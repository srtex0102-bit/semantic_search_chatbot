import { createClient, type SupabaseClient } from "@supabase/supabase-js";
import type { Database } from "@/integrations/supabase/types";

export type AppSupabaseClient = SupabaseClient<Database>;
export type TenantRole = Database["public"]["Enums"]["tenant_role"];

export interface AuthContext {
  userId: string;
  email: string | null;
  token: string;
  client: AppSupabaseClient;
}

const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

export function isValidUuid(value: unknown): value is string {
  return typeof value === "string" && UUID_RE.test(value);
}

function isNewSupabaseApiKey(value: string): boolean {
  return value.startsWith("sb_publishable_") || value.startsWith("sb_secret_");
}

function createScopedFetch(supabaseKey: string): typeof fetch {
  return (input, init) => {
    const headers = new Headers(
      typeof Request !== "undefined" && input instanceof Request ? input.headers : undefined,
    );
    if (init?.headers) {
      new Headers(init.headers).forEach((value, key) => headers.set(key, value));
    }
    if (
      isNewSupabaseApiKey(supabaseKey) &&
      headers.get("Authorization") === `Bearer ${supabaseKey}`
    ) {
      headers.delete("Authorization");
    }
    headers.set("apikey", supabaseKey);
    return fetch(input, { ...init, headers });
  };
}

export function isMissingSchemaError(error: { code?: string | null; message?: string | null } | null | undefined): boolean {
  if (!error) return false;
  const code = error.code ?? "";
  const msg = error.message ?? "";
  return (
    code === "PGRST205" ||
    code === "PGRST202" ||
    code === "42P01" ||
    code === "42883" ||
    /could not find the (table|function)/i.test(msg)
  );
}

export function missingSchemaError(_supabaseUrl?: string): HttpError {
  return new HttpError(
    503,
    "schema_not_initialized",
    "Scout's event database isn't reachable right now. Refresh the page in a moment; if this keeps happening, the app may be connected to the wrong backend.",
  );
}

export function getSupabasePublicConfig(): { url: string; publishableKey: string } {
  const url =
    process.env["VITE_SUPABASE_URL"] ||
    process.env["SUPABASE_URL"] ||
    (import.meta.env?.["VITE_SUPABASE_URL"] as string | undefined) ||
    "";
  const publishableKey =
    process.env["VITE_SUPABASE_PUBLISHABLE_KEY"] ||
    process.env["SUPABASE_PUBLISHABLE_KEY"] ||
    (import.meta.env?.["VITE_SUPABASE_PUBLISHABLE_KEY"] as string | undefined) ||
    "";
  if (!url || !publishableKey) {
    throw new Error("Missing VITE_SUPABASE_URL or VITE_SUPABASE_PUBLISHABLE_KEY.");
  }
  return { url, publishableKey };
}

export function createAnonSupabaseClient(): AppSupabaseClient {
  const { url, publishableKey } = getSupabasePublicConfig();
  return createClient<Database>(url, publishableKey, {
    global: { fetch: createScopedFetch(publishableKey) },
    auth: { storage: undefined, persistSession: false, autoRefreshToken: false },
  });
}

export function createUserSupabaseClient(token: string): AppSupabaseClient {
  const { url, publishableKey } = getSupabasePublicConfig();
  return createClient<Database>(url, publishableKey, {
    global: {
      fetch: createScopedFetch(publishableKey),
      headers: { Authorization: `Bearer ${token}` },
    },
    auth: { storage: undefined, persistSession: false, autoRefreshToken: false },
  });
}

export function hasAdminCredentials(): boolean {
  return Boolean(process.env["SUPABASE_URL"] && process.env["SUPABASE_SERVICE_ROLE_KEY"]);
}

export async function getAdminSupabaseClientOptional(): Promise<AppSupabaseClient | null> {
  if (!hasAdminCredentials()) return null;
  try {
    const { supabaseAdmin } = await import("@/integrations/supabase/client.server");
    return supabaseAdmin;
  } catch {
    return null;
  }
}

export function extractBearerToken(request: Request): string | null {
  const auth = request.headers.get("authorization") ?? "";
  if (!auth.startsWith("Bearer ")) return null;
  const token = auth.slice(7).trim();
  if (!token || token.split(".").length !== 3) return null;
  return token;
}

/**
 * Resolve optional user auth context from Bearer JWT or @supabase/ssr cookies.
 * Returns null for guests (never throws on missing token); throws 401 on invalid JWT.
 */
export async function resolveAuthContext(
  request: Request,
  options: { required?: boolean } = {},
): Promise<AuthContext | null> {
  const token = extractBearerToken(request);
  if (!token) {
    // Fallback to @supabase/ssr cookie session when present
    if (request.headers.get("cookie")) {
      try {
        const { createClient: createSsrClient } = await import("@/lib/server");
        const { supabase } = createSsrClient(request);
        const { data: claimsData } = await supabase.auth.getClaims();
        const sub = claimsData?.claims?.sub;
        if (sub) {
          const emailClaim =
            typeof claimsData.claims["email"] === "string" ? claimsData.claims["email"] : null;
          return {
            userId: sub,
            email: emailClaim,
            token: "",
            client: supabase as unknown as AppSupabaseClient,
          };
        }
      } catch {
        // ignore cookie parse issues for guests
      }
    }
    if (options.required) {
      throw new HttpError(401, "unauthorized", "Authentication is required for this operation.");
    }
    return null;
  }

  const client = createUserSupabaseClient(token);
  const { data: claimsData, error: claimsErr } = await client.auth.getClaims(token);
  if (!claimsErr && claimsData?.claims?.sub) {
    const emailClaim =
      typeof claimsData.claims["email"] === "string" ? claimsData.claims["email"] : null;
    return {
      userId: claimsData.claims.sub,
      email: emailClaim,
      token,
      client,
    };
  }

  // Fallback check via getUser for tokens where getClaims isn't locally verifiable
  const { data: userData, error: userErr } = await client.auth.getUser(token);
  if (userErr || !userData?.user?.id) {
    if (options.required) {
      throw new HttpError(401, "unauthorized", "Invalid or expired authentication token.");
    }
    return null;
  }
  return {
    userId: userData.user.id,
    email: userData.user.email ?? null,
    token,
    client,
  };
}

export async function assertWorkspaceAccess(
  client: AppSupabaseClient,
  userId: string,
  workspaceId: string,
  minRole?: TenantRole,
): Promise<TenantRole> {
  if (!isValidUuid(workspaceId)) {
    throw new HttpError(400, "invalid_workspace", "Invalid workspace ID format.");
  }
  const { data, error } = await client
    .from("tenant_members")
    .select("role")
    .eq("tenant_id", workspaceId)
    .eq("user_id", userId)
    .maybeSingle();

  if (error || !data) {
    throw new HttpError(403, "forbidden", "You do not have access to this workspace.");
  }

  const roleOrder: Record<TenantRole, number> = {
    owner: 0,
    admin: 1,
    editor: 2,
    member: 3,
    viewer: 4,
  };

  if (minRole && roleOrder[data.role] > roleOrder[minRole]) {
    throw new HttpError(403, "insufficient_role", `Requires at least ${minRole} workspace role.`);
  }
  return data.role;
}

export async function isPlatformAdmin(client: AppSupabaseClient, userId: string): Promise<boolean> {
  const { data } = await client
    .from("user_roles")
    .select("role")
    .eq("user_id", userId)
    .eq("role", "platform_admin")
    .maybeSingle();
  return Boolean(data);
}

export class HttpError extends Error {
  readonly status: number;
  readonly code: string;
  constructor(status: number, code: string, message: string) {
    super(message);
    this.status = status;
    this.code = code;
  }
}

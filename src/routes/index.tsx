import { useCallback, useEffect, useRef, useState } from "react";
import { createFileRoute } from "@tanstack/react-router";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import {
  AlertCircle,
  ArrowUp,
  Bookmark,
  Check,
  Database,
  FileText,
  GitBranch,
  LayoutDashboard,
  RefreshCw,
  RotateCcw,
  SlidersHorizontal,
  Settings2,
  Sparkles,
  Square,
  TableProperties,
  ThumbsDown,
  ThumbsUp,
} from "lucide-react";
import { toast } from "sonner";
import scoutLogo from "@/assets/scout-logo.png";
import { supabase } from "@/integrations/supabase/client";
import { AppShell } from "@/components/AppShell";
import {
  Conversation,
  ConversationContent,
  ConversationScrollButton,
} from "@/components/ai-elements/conversation";
import { Message, MessageContent, MessageResponse } from "@/components/ai-elements/message";
import { Shimmer } from "@/components/ai-elements/shimmer";
import { DynamicUISpecRenderer } from "@/components/dynamic-ui/ComponentRegistry";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Textarea } from "@/components/ui/textarea";
import {
  apiFetch,
  sendChatMessageStream,
  submitSearchFeedback,
  type ChatTurnPartial,
  type ChatTurnResult,
} from "@/lib/client/api";
import { loadBrowserChatSession, runBrowserChatTurn } from "@/lib/client/browser-chat";
import {
  BUILT_IN_MODELS,
  hasCompletedModelSettingsSetup,
  useModelSettings,
} from "@/lib/client/model-settings";
import {
  turnResultToAssistantMessage,
  useWorkspace,
  type LocalChatMessage,
} from "@/lib/client/workspace-context";
import { describeFilters } from "@/lib/events/filters";
import type { ConversationContext, SearchFilters, SearchMode } from "@/lib/events/types";
import type { AllowedFormAction, DataTableConfig, UIResponse } from "@/lib/ui/schema";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [{ title: "Scout — Intelligent Semantic Search & Dynamic AI Workspace" }],
  }),
  component: ChatWorkspacePage,
});

const STARTER_PROMPTS = [
  {
    icon: LayoutDashboard,
    label: "Analytics Dashboard",
    prompt:
      "Find all online React and Next.js events next month, compare them by date and category, and show me an analytics dashboard.",
  },
  {
    icon: Sparkles,
    label: "Semantic Discovery",
    prompt: "Find online AI and LLM events for developers in December",
  },
  {
    icon: TableProperties,
    label: "Side-by-Side Compare",
    prompt: "Compare upcoming TypeScript and React workshops by date and format",
  },
  {
    icon: GitBranch,
    label: "Search Flowchart",
    prompt: "Show how semantic search works as a flowchart",
  },
  {
    icon: SlidersHorizontal,
    label: "Dynamic Filter Form",
    prompt: "Create a form to filter events",
  },
  {
    icon: FileText,
    label: "Architecture Diagram",
    prompt: "Draw the complete architecture of this semantic search application",
  },
];

function isChatServerUnavailable(err: unknown): boolean {
  if (err instanceof TypeError) return true;
  const message = err instanceof Error ? err.message : String(err);
  return /Chat failed \((404|500|502|503|504)\)|failed to fetch|networkerror|load failed/i.test(
    message,
  );
}

function ChatWorkspacePage() {
  const queryClient = useQueryClient();
  const {
    settings,
    update: updateModelSettings,
    openRouterReady,
    ready: modelSettingsReady,
  } = useModelSettings();
  const {
    user,
    activeWorkspaceId,
    localSessions,
    saveLocalSession,
    pendingChatPrompt,
    setPendingChatPrompt,
  } = useWorkspace();

  const [sessionId, setSessionId] = useState<string | null>(null);
  const [sessionSource, setSessionSource] = useState<"remote" | "local" | null>(null);
  const [sessionSaved, setSessionSaved] = useState(false);
  const [messages, setMessages] = useState<LocalChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [mode, setMode] = useState<SearchMode>("hybrid");
  const [modelSettingsOpen, setModelSettingsOpen] = useState(false);
  const [modelSetupComplete, setModelSetupComplete] = useState(false);
  const [loadingStage, setLoadingStage] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [activeContext, setActiveContext] = useState<ConversationContext | null>(null);
  const [activeTableConfig, setActiveTableConfig] = useState<DataTableConfig | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const model =
    settings.provider === "openrouter" ? settings.openRouterModel : settings.lovableModel;
  const modelOptions =
    settings.provider === "openrouter"
      ? settings.openRouterModels.map((item) => ({ id: item.id, label: item.name }))
      : BUILT_IN_MODELS;

  useEffect(() => {
    if (hasCompletedModelSettingsSetup()) setModelSetupComplete(true);
  }, []);

  const { data: health } = useQuery({
    queryKey: ["api-health"],
    queryFn: () =>
      apiFetch<{
        schemaReady: boolean;
        schemaError?: string;
        hasEvents?: boolean;
        eventCount?: number;
      }>("/api/v1/health"),
    staleTime: 60_000,
  });

  const loadSession = useCallback(
    async (id: string) => {
      setErrorMsg(null);
      if (user) {
        try {
          const [sessRes, msgRes] = await Promise.all([
            apiFetch<{ session: { id: string; is_saved: boolean } }>(`/api/v1/chat/sessions/${id}`),
            apiFetch<{
              messages: Array<{
                id: string;
                role: "user" | "assistant";
                content: string;
                parts: unknown;
                context: ConversationContext | null;
                created_at: string;
              }>;
            }>(`/api/v1/chat/sessions/${id}/messages`),
          ]);
          setSessionId(id);
          setSessionSource("remote");
          setSessionSaved(sessRes.session.is_saved);
          const hydrated: LocalChatMessage[] = msgRes.messages.map((m) => {
            const parts = Array.isArray(m.parts) ? m.parts : [];
            const uiPart = parts.find((p): p is { type: "ui_response"; spec: UIResponse } =>
              Boolean(p && typeof p === "object" && "type" in p && p.type === "ui_response"),
            );
            return {
              id: m.id,
              role: m.role,
              content: m.content,
              ...(uiPart?.spec ? { uiSpec: uiPart.spec } : {}),
              ...(m.context ? { context: m.context } : {}),
              createdAt: m.created_at,
            };
          });
          setMessages(hydrated);
          const lastCtx = [...hydrated].reverse().find((m) => m.context)?.context ?? null;
          setActiveContext(lastCtx);
        } catch (err) {
          try {
            const browserSession = await loadBrowserChatSession(id);
            if (browserSession) {
              setSessionId(id);
              setSessionSource("remote");
              setSessionSaved(browserSession.session.is_saved);
              const hydrated: LocalChatMessage[] = browserSession.messages.map((m) => {
                const parts = Array.isArray(m.parts) ? m.parts : [];
                const uiPart = parts.find((p): p is { type: "ui_response"; spec: UIResponse } =>
                  Boolean(p && typeof p === "object" && "type" in p && p["type"] === "ui_response"),
                );
                return {
                  id: m.id,
                  role: m.role as "user" | "assistant",
                  content: m.content,
                  ...(uiPart?.spec ? { uiSpec: uiPart.spec } : {}),
                  ...(m.context ? { context: m.context as unknown as ConversationContext } : {}),
                  createdAt: m.created_at,
                };
              });
              setMessages(hydrated);
              const lastCtx = [...hydrated].reverse().find((m) => m.context)?.context ?? null;
              setActiveContext(lastCtx);
              return;
            }
          } catch {
            // Fall back to the browser copy if Supabase is unavailable.
          }
          const local = localSessions.find((s) => s.id === id);
          if (local) {
            setSessionId(local.id);
            setSessionSource("local");
            setSessionSaved(local.isSaved);
            setMessages(local.messages);
            const lastCtx = [...local.messages].reverse().find((m) => m.context)?.context ?? null;
            setActiveContext(lastCtx);
          } else {
            toast.error(err instanceof Error ? err.message : "Unable to load conversation");
          }
        }
      } else {
        const local = localSessions.find((s) => s.id === id);
        if (local) {
          setSessionId(local.id);
          setSessionSource("local");
          setSessionSaved(local.isSaved);
          setMessages(local.messages);
          const lastCtx = [...local.messages].reverse().find((m) => m.context)?.context ?? null;
          setActiveContext(lastCtx);
        }
      }
    },
    [localSessions, user],
  );

  // Sync URL ?session=<id> if navigated from another view
  useEffect(() => {
    if (typeof window === "undefined") return;
    const sid = new URLSearchParams(window.location.search).get("session");
    if (sid && sid !== sessionId) {
      void loadSession(sid);
    }
  }, [loadSession, sessionId]);

  const startNewChat = useCallback(() => {
    abortRef.current?.abort();
    if (typeof window !== "undefined") {
      const url = new URL(window.location.href);
      if (url.searchParams.has("session")) {
        url.searchParams.delete("session");
        window.history.replaceState(window.history.state, "", url);
      }
    }
    setSessionId(null);
    setSessionSource(null);
    setSessionSaved(false);
    setMessages([]);
    setActiveContext(null);
    setActiveTableConfig(null);
    setErrorMsg(null);
    setLoadingStage(null);
  }, []);

  const submitPrompt = useCallback(
    async (text: string, explicitFilters?: Partial<SearchFilters>) => {
      const clean = text.trim();
      if (!clean || loadingStage || !modelSettingsReady) return;
      if (!modelSetupComplete) {
        setModelSettingsOpen(true);
        return;
      }
      if (settings.provider === "openrouter" && !openRouterReady) {
        setErrorMsg("Connect an OpenRouter key in AI model settings before sending a message.");
        setModelSettingsOpen(true);
        return;
      }

      setErrorMsg(null);
      setInput("");
      const userMsg: LocalChatMessage = {
        id: crypto.randomUUID(),
        role: "user",
        content: clean,
        createdAt: new Date().toISOString(),
      };
      const nextMsgs = [...messages, userMsg];
      const streamMessageId = crypto.randomUUID();
      setMessages([
        ...nextMsgs,
        {
          id: streamMessageId,
          role: "assistant",
          content: "",
          streaming: true,
          createdAt: new Date().toISOString(),
        },
      ]);
      setLoadingStage("Understanding your question");

      const controller = new AbortController();
      abortRef.current = controller;

      try {
        const payload = {
          message: clean,
          sessionId: user && sessionSource === "remote" ? sessionId : null,
          workspaceId: activeWorkspaceId,
          mode,
          model,
          provider: settings.provider,
          chatMode: settings.chatMode,
          history: messages.slice(-12).map((m) => ({ role: m.role, content: m.content })),
          ...(explicitFilters ? { explicitFilters } : {}),
          context: activeContext,
          prevTableConfig: activeTableConfig,
        };
        const callbacks = {
          onStatus: (stage: string) => setLoadingStage(stage),
          onPartial: (partial: ChatTurnPartial) => {
            setMessages((current) =>
              current.map((message) => {
                if (message.id !== streamMessageId) return message;
                const { uiSpec, ...partialMetadata } = partial;
                return { ...message, ...partialMetadata, ...(uiSpec ? { uiSpec } : {}) };
              }),
            );
          },
          onDelta: (delta: string) => {
            setMessages((current) =>
              current.map((message) =>
                message.id === streamMessageId
                  ? { ...message, content: message.content + delta }
                  : message,
              ),
            );
          },
          signal: controller.signal,
          openRouterKey: settings.provider === "openrouter" ? settings.openRouterKey : null,
        };

        let result: ChatTurnResult;
        try {
          result = await sendChatMessageStream(payload, callbacks);
        } catch (err) {
          if (controller.signal.aborted || !isChatServerUnavailable(err)) throw err;
          setLoadingStage("Chat server unavailable — switching to browser mode");
          result = await runBrowserChatTurn(
            { ...payload, userId: user?.id ?? null, openRouterKey: callbacks.openRouterKey },
            callbacks,
          );
        }

        const assistantMsg = turnResultToAssistantMessage(result);
        const updatedMsgs = [...nextMsgs, assistantMsg];
        setMessages((current) => [
          ...current.filter((message) => message.id !== streamMessageId),
          assistantMsg,
        ]);
        setActiveContext(result.context);

        const tableComp = result.uiSpec?.components.find((c) => c.type === "data_table");
        if (tableComp && tableComp.type === "data_table") {
          setActiveTableConfig(tableComp.props.config);
        }

        const resolvedSid =
          result.sessionId ?? (sessionSource === "local" ? sessionId : null) ?? crypto.randomUUID();
        setSessionId(resolvedSid);
        setSessionSource(user && result.sessionId ? "remote" : "local");
        // Keep the open conversation in the address bar so a page refresh reopens it.
        const url = new URL(window.location.href);
        url.searchParams.set("session", resolvedSid);
        window.history.replaceState(window.history.state, "", url);

        if (user && result.sessionId) {
          void queryClient.invalidateQueries({ queryKey: ["chat-sessions"] });
        }
        if (!user || !result.sessionId) {
          saveLocalSession({
            id: resolvedSid,
            title: nextMsgs[0]?.content.slice(0, 70) || clean.slice(0, 70),
            isSaved: sessionSaved,
            workspaceId: activeWorkspaceId,
            messages: updatedMsgs,
            updatedAt: new Date().toISOString(),
          });
        }
      } catch (err) {
        setMessages((current) => current.filter((message) => message.id !== streamMessageId));
        if (controller.signal.aborted) {
          setErrorMsg("Generation stopped.");
        } else {
          setErrorMsg(err instanceof Error ? err.message : "Failed to complete request.");
        }
      } finally {
        setLoadingStage(null);
        abortRef.current = null;
      }
    },
    [
      activeContext,
      activeTableConfig,
      activeWorkspaceId,
      loadingStage,
      messages,
      mode,
      modelSetupComplete,
      modelSettingsReady,
      model,
      openRouterReady,
      queryClient,
      saveLocalSession,
      sessionId,
      sessionSource,
      sessionSaved,
      settings,
      user,
    ],
  );

  // Handle cross-page prompt jumps ("Find Similar in Chat" / "Template: React Analytics")
  useEffect(() => {
    if (pendingChatPrompt && !loadingStage && modelSettingsReady && modelSetupComplete) {
      const p = pendingChatPrompt;
      setPendingChatPrompt(null);
      void submitPrompt(p);
    }
  }, [
    pendingChatPrompt,
    setPendingChatPrompt,
    loadingStage,
    modelSettingsReady,
    modelSetupComplete,
    submitPrompt,
  ]);

  const handleDynamicFormSubmit = (action: AllowedFormAction, values: Record<string, unknown>) => {
    if (action === "filter_search") {
      const q = typeof values["query"] === "string" ? values["query"].trim() : "";
      const formats = Array.isArray(values["formats"])
        ? (values["formats"] as SearchFilters["formats"])
        : [];
      const categories = Array.isArray(values["categories"])
        ? (values["categories"] as string[])
        : [];
      const technologies = Array.isArray(values["technologies"])
        ? (values["technologies"] as string[])
        : [];
      const city =
        typeof values["city"] === "string" && values["city"] && values["city"] !== "__any__"
          ? values["city"]
          : null;
      const dateStart =
        typeof values["dateStart"] === "string" && values["dateStart"] ? values["dateStart"] : null;
      const dateEnd =
        typeof values["dateEnd"] === "string" && values["dateEnd"] ? values["dateEnd"] : null;

      const summaryBits = [
        q || "all matching events",
        technologies.length ? `for ${technologies.join(", ")}` : "",
        formats.length ? `(${formats.join(", ")})` : "",
        values["freeOnly"] ? "free" : "",
      ]
        .filter(Boolean)
        .join(" ");

      void submitPrompt(`Find ${summaryBits}`, {
        formats,
        categories,
        technologies,
        city,
        dateStart,
        dateEnd,
      });
    } else {
      toast.success(`Executed ${action} configuration`);
    }
  };

  const toggleSaveSession = async () => {
    const nextSaved = !sessionSaved;
    setSessionSaved(nextSaved);
    if (user && sessionId && sessionSource === "remote") {
      try {
        await apiFetch(`/api/v1/chat/sessions/${sessionId}`, {
          method: "PATCH",
          body: JSON.stringify({ is_saved: nextSaved }),
        });
        await queryClient.invalidateQueries({ queryKey: ["chat-sessions"] });
      } catch {
        const { data, error } = await supabase
          .from("chat_sessions")
          .update({ is_saved: nextSaved })
          .eq("id", sessionId)
          .select("id")
          .maybeSingle();
        if (error || !data) {
          setSessionSaved(!nextSaved);
          toast.error(error?.message ?? "Unable to update this conversation.");
          return;
        }
        await queryClient.invalidateQueries({ queryKey: ["chat-sessions"] });
      }
    } else if (sessionId) {
      const existing = localSessions.find((s) => s.id === sessionId);
      saveLocalSession({
        id: sessionId,
        title:
          existing?.title ??
          messages.find((m) => m.role === "user")?.content.slice(0, 70) ??
          "New conversation",
        isSaved: nextSaved,
        workspaceId: existing?.workspaceId ?? activeWorkspaceId,
        messages,
        updatedAt: new Date().toISOString(),
      });
    }
    toast.success(nextSaved ? "Conversation saved" : "Removed from saved");
  };

  const regenerateLast = () => {
    const lastUser = [...messages].reverse().find((m) => m.role === "user");
    if (lastUser) void submitPrompt(lastUser.content);
  };

  return (
    <AppShell
      onSelectSession={(id) => void loadSession(id)}
      activeSessionId={sessionId}
      onNewChat={startNewChat}
      modelSettingsOpen={modelSettingsOpen}
      onModelSettingsOpenChange={setModelSettingsOpen}
      onModelSettingsSetupComplete={() => setModelSetupComplete(true)}
    >
      <div className="flex h-[calc(100vh-3rem)] flex-col">
        {/* Session Status Bar */}
        {(messages.length > 0 || activeContext) && (
          <div className="flex flex-wrap items-center justify-between gap-2 border-b bg-muted/20 px-4 py-1.5 text-xs">
            <div className="flex flex-wrap items-center gap-2">
              {activeContext && (
                <>
                  <Badge variant="secondary" className="gap-1 text-[11px]">
                    <Database className="size-3" />
                    {activeContext.resultIds.length} active record
                    {activeContext.resultIds.length === 1 ? "" : "s"} in context
                  </Badge>
                  {describeFilters(activeContext.filters).map((f) => (
                    <Badge key={f} variant="outline" className="text-[11px]">
                      {f}
                    </Badge>
                  ))}
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    className="h-6 px-2 text-[11px] text-muted-foreground"
                    onClick={() => setActiveContext(null)}
                  >
                    Clear context
                  </Button>
                </>
              )}
            </div>

            <div className="flex items-center gap-1.5">
              {sessionId && (
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  className="h-6 gap-1 px-2 text-[11px]"
                  onClick={() => void toggleSaveSession()}
                >
                  <Bookmark
                    className={`size-3 ${sessionSaved ? "fill-primary text-primary" : ""}`}
                  />
                  {sessionSaved ? "Saved" : "Save chat"}
                </Button>
              )}
              {messages.length > 0 && (
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  className="h-6 gap-1 px-2 text-[11px]"
                  onClick={startNewChat}
                >
                  <RotateCcw className="size-3" /> New chat
                </Button>
              )}
            </div>
          </div>
        )}

        {/* Main Conversation Stream */}
        <Conversation className="flex-1">
          <ConversationContent className="mx-auto w-full max-w-5xl px-4 py-6">
            {health && !health.schemaReady && (
              <Alert>
                <AlertCircle className="size-4" />
                <AlertDescription className="text-xs">
                  The Scout database schema is not ready. Confirm this app is connected to the
                  intended Supabase project and ask Lovable to provision or repair the managed
                  schema. Live search needs the existing event tables, search RPCs, and RLS
                  policies.
                  {health.schemaError ? ` ${health.schemaError}` : ""}
                </AlertDescription>
              </Alert>
            )}
            {messages.length === 0 ? (
              <div className="mx-auto flex max-w-3xl flex-col items-center space-y-6 py-8 text-center">
                <img
                  src={scoutLogo}
                  alt="Scout Workspace"
                  className="size-14 rounded-2xl border shadow-sm"
                />
                <div className="space-y-2">
                  <Badge variant="outline" className="gap-1.5 px-3 py-0.5 text-xs">
                    <Sparkles className="size-3 text-primary" />
                    Semantic Search + Controlled Dynamic UI
                  </Badge>
                  <h1 className="text-2xl font-bold tracking-tight sm:text-3xl">
                    Ask in natural language. Get grounded data &amp; live interfaces.
                  </h1>
                  <p className="mx-auto max-w-xl text-sm text-muted-foreground">
                    Scout retrieves authorized events via hybrid vector + keyword + metadata search,
                    explains every match, and composes validated tables, charts, diagrams, forms,
                    and dashboards.
                  </p>
                </div>

                <div className="grid w-full grid-cols-1 gap-2.5 text-left sm:grid-cols-2 lg:grid-cols-3">
                  {STARTER_PROMPTS.map((item) => {
                    const Icon = item.icon;
                    return (
                      <button
                        key={item.label}
                        type="button"
                        onClick={() => void submitPrompt(item.prompt)}
                        className="group flex flex-col gap-1.5 rounded-xl border bg-card p-3.5 text-left transition-all hover:border-primary/40 hover:shadow-sm"
                      >
                        <div className="flex items-center gap-2 text-xs font-semibold text-foreground group-hover:text-primary">
                          <Icon className="size-4 text-primary" />
                          {item.label}
                        </div>
                        <p className="line-clamp-2 text-xs text-muted-foreground">{item.prompt}</p>
                      </button>
                    );
                  })}
                </div>
              </div>
            ) : (
              messages.map((msg, idx) => (
                <Message key={msg.id} from={msg.role} className="max-w-full">
                  {msg.role === "user" ? (
                    <MessageContent className="max-w-[85%] rounded-2xl bg-primary px-4 py-2.5 text-primary-foreground shadow-xs">
                      <p className="whitespace-pre-wrap text-sm">{msg.content}</p>
                    </MessageContent>
                  ) : (
                    <div className="w-full space-y-4">
                      {msg.securityNotice && (
                        <Alert>
                          <AlertCircle className="size-4" />
                          <AlertDescription className="text-xs">
                            {msg.securityNotice}
                          </AlertDescription>
                        </Alert>
                      )}
                      {msg.runtimeNotice && (
                        <Alert>
                          <AlertCircle className="size-4" />
                          <AlertDescription className="text-xs">
                            {msg.runtimeNotice}
                          </AlertDescription>
                        </Alert>
                      )}

                      <div className="flex flex-wrap items-center gap-2 text-[11px] text-muted-foreground">
                        <Badge variant="outline" className="text-[10px]">
                          {msg.semanticAvailable
                            ? "Hybrid Semantic + Keyword"
                            : "Keyword + Metadata"}
                        </Badge>
                        {msg.executionMode === "browser" && (
                          <Badge variant="secondary" className="text-[10px]">
                            Browser mode
                          </Badge>
                        )}
                        {msg.lowConfidence && (
                          <Badge variant="secondary" className="text-[10px]">
                            Low confidence — partial matches
                          </Badge>
                        )}
                        {msg.latencyMs !== undefined && (
                          <span className="tabular-nums">{msg.latencyMs} ms</span>
                        )}
                        {msg.modelUsed && <span>· {msg.modelUsed}</span>}
                      </div>

                      <MessageContent className="w-full max-w-none text-sm leading-relaxed">
                        {msg.content ? (
                          <MessageResponse>{msg.content}</MessageResponse>
                        ) : msg.streaming ? (
                          <Shimmer className="text-sm">Thinking…</Shimmer>
                        ) : null}
                        {msg.streaming && msg.content && (
                          <span className="ml-1 inline-block size-1.5 animate-pulse rounded-full bg-primary align-middle" />
                        )}
                      </MessageContent>

                      {msg.uiSpec && (
                        <DynamicUISpecRenderer
                          spec={msg.uiSpec}
                          events={msg.events ?? []}
                          messageId={msg.id}
                          query={messages[idx - 1]?.content ?? ""}
                          onFormSubmit={handleDynamicFormSubmit}
                        />
                      )}

                      {/* Follow-up suggestions & Feedback */}
                      <div className="flex flex-wrap items-center justify-between gap-2 pt-1">
                        <div className="flex flex-wrap gap-1.5">
                          {(msg.followUpSuggestions ?? []).map((sug) => (
                            <Button
                              key={sug}
                              type="button"
                              variant="outline"
                              size="sm"
                              className="h-7 rounded-full px-3 text-xs"
                              disabled={Boolean(loadingStage)}
                              onClick={() => void submitPrompt(sug)}
                            >
                              {sug}
                            </Button>
                          ))}
                        </div>

                        <div className="flex items-center gap-1">
                          <Button
                            type="button"
                            size="sm"
                            variant="ghost"
                            className="h-7 gap-1 px-2 text-xs text-muted-foreground"
                            onClick={() => {
                              void submitSearchFeedback({
                                messageId: msg.id,
                                query: messages[idx - 1]?.content ?? "",
                                rating: "useful",
                              });
                              toast.success("Marked answer helpful");
                            }}
                          >
                            <ThumbsUp className="size-3" /> Helpful
                          </Button>
                          <Button
                            type="button"
                            size="sm"
                            variant="ghost"
                            className="h-7 gap-1 px-2 text-xs text-muted-foreground"
                            onClick={() => {
                              void submitSearchFeedback({
                                messageId: msg.id,
                                query: messages[idx - 1]?.content ?? "",
                                rating: "irrelevant",
                              });
                              toast.success("Feedback recorded");
                            }}
                          >
                            <ThumbsDown className="size-3" />
                          </Button>
                          {idx === messages.length - 1 && !msg.streaming && !loadingStage && (
                            <Button
                              type="button"
                              size="sm"
                              variant="ghost"
                              className="h-7 gap-1 px-2 text-xs text-muted-foreground"
                              disabled={Boolean(loadingStage)}
                              onClick={regenerateLast}
                            >
                              <RefreshCw className="size-3" /> Regenerate
                            </Button>
                          )}
                        </div>
                      </div>
                    </div>
                  )}
                </Message>
              ))
            )}

            {loadingStage && (
              <div className="flex items-center gap-3 rounded-xl border bg-muted/30 px-4 py-3 text-sm">
                <Check className="size-4 animate-pulse text-primary" />
                <Shimmer className="text-sm font-medium">{loadingStage}</Shimmer>
              </div>
            )}

            {errorMsg && (
              <Alert variant="destructive">
                <AlertCircle className="size-4" />
                <AlertDescription className="flex items-center justify-between text-xs">
                  <span>{errorMsg}</span>
                  <Button type="button" size="sm" variant="outline" onClick={regenerateLast}>
                    Retry
                  </Button>
                </AlertDescription>
              </Alert>
            )}
          </ConversationContent>
          <ConversationScrollButton />
        </Conversation>

        {/* Bottom Prompt Composer */}
        <div className="border-t bg-background/95 p-3 backdrop-blur">
          <form
            onSubmit={(e) => {
              e.preventDefault();
              void submitPrompt(input);
            }}
            className="mx-auto max-w-4xl space-y-2 rounded-2xl border bg-card p-2.5 shadow-sm focus-within:ring-2 focus-within:ring-ring"
          >
            <Textarea
              aria-label="Ask Scout about events, or request a table, chart, diagram, form, or dashboard"
              placeholder="Ask anything — e.g. 'Find all online React and Next.js events next month, compare them by date and category, and show me an analytics dashboard'"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  void submitPrompt(input);
                }
              }}
              rows={2}
              className="min-h-[48px] resize-none border-0 bg-transparent p-1 text-sm shadow-none focus-visible:ring-0"
            />

            <div className="flex flex-wrap items-center justify-between gap-2 pt-1">
              <div className="flex flex-wrap items-center gap-2">
                <Select value={mode} onValueChange={(v) => setMode(v as SearchMode)}>
                  <SelectTrigger className="h-7 w-[118px] text-xs" aria-label="Search mode">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="hybrid">Hybrid Search</SelectItem>
                    <SelectItem value="semantic">Semantic Only</SelectItem>
                    <SelectItem value="keyword">Keyword Only</SelectItem>
                  </SelectContent>
                </Select>

                <Select
                  value={settings.chatMode}
                  onValueChange={(value) =>
                    updateModelSettings({ chatMode: value as typeof settings.chatMode })
                  }
                >
                  <SelectTrigger className="h-7 w-[122px] text-xs" aria-label="Assistant mode">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="auto">Auto mode</SelectItem>
                    <SelectItem value="search">Event search</SelectItem>
                    <SelectItem value="chat">Assistant chat</SelectItem>
                  </SelectContent>
                </Select>

                <Select
                  value={model}
                  onValueChange={(value) =>
                    updateModelSettings(
                      settings.provider === "openrouter"
                        ? { openRouterModel: value }
                        : { lovableModel: value },
                    )
                  }
                >
                  <SelectTrigger className="h-7 w-[155px] text-xs" aria-label="AI model">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {modelOptions.map((m) => (
                      <SelectItem key={m.id} value={m.id}>
                        {m.label}
                      </SelectItem>
                    ))}
                    {settings.provider === "openrouter" && modelOptions.length === 0 && (
                      <SelectItem value="openrouter/free">OpenRouter Free Router</SelectItem>
                    )}
                  </SelectContent>
                </Select>
                <Button
                  type="button"
                  size="icon"
                  variant="ghost"
                  className="size-7"
                  aria-label="Change AI provider or model settings"
                  onClick={() => setModelSettingsOpen(true)}
                >
                  <Settings2 className="size-3.5" />
                </Button>
              </div>

              <div className="flex items-center gap-1.5">
                {loadingStage ? (
                  <Button
                    type="button"
                    size="sm"
                    variant="destructive"
                    className="h-8 gap-1.5 px-3 text-xs"
                    onClick={() => abortRef.current?.abort()}
                  >
                    <Square className="size-3 fill-current" /> Stop
                  </Button>
                ) : (
                  <Button
                    type="submit"
                    size="sm"
                    disabled={!input.trim()}
                    className="h-8 gap-1.5 px-3 text-xs"
                  >
                    <ArrowUp className="size-3.5" /> Send
                  </Button>
                )}
              </div>
            </div>
          </form>
        </div>
      </div>
    </AppShell>
  );
}

import { sanitizeFilters, mergeFilters } from "@/lib/events/filters";
import type { ConversationContext, EventFormat, SearchFilters } from "@/lib/events/types";
import type { ChartDimension } from "@/lib/events/aggregate";
import type { ChartType, DiagramType } from "@/lib/ui/schema";
import type { TableRefinementCommand } from "@/lib/ui/validator";

export type WorkspaceIntent =
  | "search_events"
  | "compare_events"
  | "analytics_chart"
  | "build_dashboard"
  | "generate_diagram"
  | "generate_form"
  | "refine_table"
  | "timeline_view"
  | "kanban_view"
  | "checklist_view"
  | "export_report"
  | "explain_search"
  | "clarify";

export interface ParsedIntent {
  intent: WorkspaceIntent;
  searchText: string;
  topics: string[];
  audience: string[];
  filters: SearchFilters;
  isFreeOnly: boolean;
  isFollowUp: boolean;
  reusePreviousResults: boolean;
  clearPreviousFilters: boolean;
  requestedChartType?: ChartType;
  requestedDimension?: ChartDimension;
  requestedDiagramType?: DiagramType;
  requestedTable?: boolean;
  requestedCards?: boolean;
  tableCommand?: TableRefinementCommand;
  confidence: number;
}

const KNOWN_TOPICS: Array<{ label: string; patterns: RegExp[] }> = [
  { label: "React", patterns: [/\breact\b(?!\s+native|\s+router)/i, /\breact\.?js\b/i] },
  { label: "React Native", patterns: [/\breact\s+native\b/i, /\bexpo\b/i] },
  { label: "React Router", patterns: [/\breact\s+router\b/i, /\bremix\b/i] },
  { label: "Next.js", patterns: [/\bnext\.?js\b/i, /\bnextjs\b/i] },
  { label: "TypeScript", patterns: [/\btypescript\b/i, /\bts\b/i] },
  { label: "JavaScript", patterns: [/\bjavascript\b/i, /\bjs\b/i, /\bnode\.?js\b/i] },
  { label: "AI", patterns: [/\bai\b/i, /\bartificial intelligence\b/i, /\bgenerative ai\b/i, /\brag\b/i, /\bagents?\b/i] },
  { label: "LLM", patterns: [/\bllms?\b/i, /\blarge language models?\b/i] },
  { label: "Machine Learning", patterns: [/\bmachine learning\b/i, /\bmlops\b/i, /\bml\b/i] },
  { label: "Python", patterns: [/\bpython\b/i, /\bpydata\b/i, /\basyncio\b/i] },
  { label: "Kubernetes", patterns: [/\bkubernetes\b/i, /\bk8s\b/i, /\bkubecon\b/i] },
  { label: "Cloud", patterns: [/\bcloud\b/i, /\baws\b/i, /\bserverless\b/i, /\bedge computing\b/i] },
  { label: "DevOps", patterns: [/\bdevops\b/i, /\bobservability\b/i, /\bsre\b/i] },
  { label: "Security", patterns: [/\bsecurity\b/i, /\bsecurecode\b/i] },
  { label: "Data Engineering", patterns: [/\bdata engineering\b/i, /\bdata mesh\b/i, /\bsql\b/i, /\bpostgresql\b/i, /\bpgvector\b/i, /\bvector\b/i] },
  { label: "Data Visualization", patterns: [/\bdata visualization\b/i, /\bd3\b/i] },
  { label: "Design Systems", patterns: [/\bdesign systems?\b/i, /\bfigma\b/i, /\bcss\b/i, /\baccessibility\b/i, /\bwcag\b/i] },
  { label: "Rust", patterns: [/\brust\b/i, /\bwebassembly\b/i, /\bwasm\b/i] },
  { label: "Go", patterns: [/\bgolang\b/i, /\bgo\s+systems\b/i] },
  { label: "Vue", patterns: [/\bvue\b/i, /\bnuxt\b/i] },
  { label: "Svelte", patterns: [/\bsvelte\b/i, /\bsveltekit\b/i] },
  { label: "Testing", patterns: [/\btesting\b/i, /\bplaywright\b/i, /\bvitest\b/i] },
  { label: "Web3", patterns: [/\bweb3\b/i, /\bsolidity\b/i, /\bethereum\b/i, /\bblockchain\b/i] },
];

const KNOWN_CITIES = [
  "Amsterdam",
  "Bangalore",
  "Berlin",
  "London",
  "New York",
  "San Francisco",
  "Singapore",
  "Toronto",
];

const KNOWN_CATEGORIES = [
  "Conference",
  "Workshop",
  "Meetup",
  "Webinar",
  "Hackathon",
  "Bootcamp",
];

const MONTH_NAMES: Record<string, number> = {
  january: 0,
  jan: 0,
  february: 1,
  feb: 1,
  march: 2,
  mar: 2,
  april: 3,
  apr: 3,
  may: 4,
  june: 5,
  jun: 5,
  july: 6,
  jul: 6,
  august: 7,
  aug: 7,
  september: 8,
  sep: 8,
  sept: 8,
  october: 9,
  oct: 9,
  november: 10,
  nov: 10,
  december: 11,
  dec: 11,
};

function monthBounds(year: number, monthZeroIdx: number): { dateStart: string; dateEnd: string } {
  const start = new Date(Date.UTC(year, monthZeroIdx, 1));
  const end = new Date(Date.UTC(year, monthZeroIdx + 1, 0));
  return {
    dateStart: start.toISOString().slice(0, 10),
    dateEnd: end.toISOString().slice(0, 10),
  };
}

function extractDateRange(query: string, now: Date): { dateStart: string | null; dateEnd: string | null } {
  const explicit = query.match(/\b(\d{4}-\d{2}-\d{2})\b/g);
  if (explicit && explicit.length >= 2) {
    return { dateStart: explicit[0] ?? null, dateEnd: explicit[1] ?? null };
  }
  if (explicit && explicit.length === 1) {
    return { dateStart: explicit[0] ?? null, dateEnd: null };
  }

  const lower = query.toLowerCase();
  const curY = now.getUTCFullYear();
  const curM = now.getUTCMonth();

  if (/\bnext month\b/.test(lower)) {
    const d = new Date(Date.UTC(curY, curM + 1, 1));
    return monthBounds(d.getUTCFullYear(), d.getUTCMonth());
  }
  if (/\bthis month\b/.test(lower)) {
    return monthBounds(curY, curM);
  }

  const monthRegex =
    /\b(january|jan|february|feb|march|mar|april|apr|may|june|jun|july|jul|august|aug|september|sep|sept|october|oct|november|nov|december|dec)(?:\s+(202\d))?\b/i;
  const mMatch = lower.match(monthRegex);
  if (mMatch && mMatch[1]) {
    const monthIdx = MONTH_NAMES[mMatch[1].toLowerCase()];
    if (monthIdx !== undefined) {
      let year = mMatch[2] ? Number(mMatch[2]) : curY;
      if (!mMatch[2] && monthIdx < curM - 1) {
        year = curY + 1;
      }
      return monthBounds(year, monthIdx);
    }
  }

  if (/\bupcoming\b|\bfuture\b/.test(lower)) {
    return { dateStart: now.toISOString().slice(0, 10), dateEnd: null };
  }

  return { dateStart: null, dateEnd: null };
}

function extractTableCommand(query: string): TableRefinementCommand | undefined {
  const lower = query.toLowerCase();
  const cmd: TableRefinementCommand = {};
  let found = false;

  if (/\bsort\s+(?:by\s+)?(?:event\s+)?date\b|\bchronological\b|\bearliest\b/.test(lower)) {
    cmd.sortBy = "startAt";
    cmd.sortDir = /\bdesc|latest|newest\b/.test(lower) ? "desc" : "asc";
    found = true;
  } else if (/\bsort\s+(?:by\s+)?(?:score|relevance)\b/.test(lower)) {
    cmd.sortBy = "score";
    cmd.sortDir = "desc";
    found = true;
  } else if (/\bsort\s+(?:by\s+)?title\b/.test(lower)) {
    cmd.sortBy = "title";
    cmd.sortDir = "asc";
    found = true;
  } else if (/\bsort\s+(?:by\s+)?category\b/.test(lower)) {
    cmd.sortBy = "category";
    cmd.sortDir = "asc";
    found = true;
  }

  const hideCols: NonNullable<TableRefinementCommand["hideColumns"]> = [];
  if (/\bhide\s+(?:the\s+)?description\b/.test(lower)) hideCols.push("description");
  if (/\bhide\s+(?:the\s+)?summary\b/.test(lower)) hideCols.push("description");
  if (/\bhide\s+(?:the\s+)?score\b/.test(lower)) hideCols.push("score");
  if (/\bhide\s+(?:the\s+)?location\b/.test(lower)) hideCols.push("location");
  if (/\bhide\s+(?:the\s+)?audience\b/.test(lower)) hideCols.push("audience");
  if (/\bhide\s+(?:the\s+)?explanation\b/.test(lower)) hideCols.push("explanation");
  if (hideCols.length) {
    cmd.hideColumns = hideCols;
    found = true;
  }

  const showCols: NonNullable<TableRefinementCommand["showColumns"]> = [];
  if (/\bshow\s+(?:the\s+)?description\b/.test(lower)) showCols.push("description");
  if (/\bshow\s+(?:the\s+)?audience\b/.test(lower)) showCols.push("audience");
  if (/\bshow\s+(?:the\s+)?explanation\b/.test(lower)) showCols.push("explanation");
  if (showCols.length) {
    cmd.showColumns = showCols;
    found = true;
  }

  const groupMatch = lower.match(/\bgroup\s+(?:events\s+)?by\s+(category|format|month)\b/);
  if (groupMatch && groupMatch[1]) {
    cmd.groupBy = groupMatch[1] as "category" | "format" | "month";
    found = true;
  }

  const topMatch = lower.match(/\b(?:top|first)\s+(\d+|ten|five|three)\b/);
  if (topMatch && topMatch[1]) {
    const wordMap: Record<string, number> = { three: 3, five: 5, ten: 10 };
    const n = wordMap[topMatch[1]] ?? Number(topMatch[1]);
    if (Number.isFinite(n) && n >= 1 && n <= 60) {
      cmd.maxRows = n;
      found = true;
    }
  }

  return found ? cmd : undefined;
}

function detectChartType(lower: string): ChartType | undefined {
  if (/\bhorizontal\s+bar\b/.test(lower)) return "horizontal_bar";
  if (/\bstacked\s+bar\b/.test(lower)) return "stacked_bar";
  if (/\bbar\s+chart\b|\bhistogram\b/.test(lower)) return "bar";
  if (/\bline\s+chart\b|\btrend\b|\bover\s+time\b/.test(lower)) return "line";
  if (/\barea\s+chart\b/.test(lower)) return "area";
  if (/\bdonut\b|\bdoughnut\b/.test(lower)) return "donut";
  if (/\bpie\s+chart\b|\bbreakdown\b/.test(lower)) return "pie";
  if (/\bscatter\b|\bcorrelation\b/.test(lower)) return "scatter";
  return undefined;
}

function detectChartDimension(lower: string): ChartDimension {
  if (/\btechnolog(?:y|ies)\b|\bstack\b|\btools?\b/.test(lower)) return "technology";
  if (/\bformat\b|\bonline\s+(?:vs|versus)\s+in[- ]person\b/.test(lower)) return "format";
  if (/\bmonth\b|\bover\s+time\b|\btrend\b|\bby\s+date\b/.test(lower)) return "month";
  if (/\bcity|cities|location\b/.test(lower)) return "city";
  if (/\baudience\b|\bwho\b/.test(lower)) return "audience";
  return "category";
}

function detectDiagramType(lower: string): DiagramType {
  if (/\bsequence\b/.test(lower)) return "sequence";
  if (/\barchitecture\b|\bsystem\b|\binfrastructure\b/.test(lower)) return "architecture";
  if (/\ber\s+diagram\b|\bentity\s+relationship\b|\bdatabase\s+schema\b|\bschema\b/.test(lower)) return "er_diagram";
  if (/\bmind\s*map\b/.test(lower)) return "mind_map";
  if (/\bdecision\s+tree\b/.test(lower)) return "decision_tree";
  if (/\buser\s+journey\b|\bjourney\b/.test(lower)) return "user_journey";
  if (/\bstate\s+diagram\b/.test(lower)) return "state";
  if (/\bprocess\b|\bpipeline\b|\bindexing\b/.test(lower)) return "process";
  return "flowchart";
}

/**
 * Deterministically parse natural-language requests into structured search
 * filters, workspace intent, UI component preferences, and follow-up merging.
 */
export function parseUserIntent(
  rawQuery: string,
  prevContext?: ConversationContext | null,
  now: Date = new Date(),
): ParsedIntent {
  const query = rawQuery.trim();
  const lower = query.toLowerCase();

  // Extract formats
  const formats: EventFormat[] = [];
  if (/\bonline\b|\bvirtual\b|\bremote\b/.test(lower)) formats.push("online");
  if (/\bin[- ]person\b|\bonsite\b|\blocal\b/.test(lower)) formats.push("in_person");
  if (/\bhybrid\b/.test(lower)) formats.push("hybrid");

  // Extract categories
  const categories = KNOWN_CATEGORIES.filter((cat) =>
    new RegExp(`\\b${cat.toLowerCase()}s?\\b`, "i").test(lower),
  );

  // Extract city
  let city: string | null = null;
  if (/\bnyc\b|\bnew york\b/i.test(lower)) city = "New York";
  else {
    for (const c of KNOWN_CITIES) {
      if (new RegExp(`\\b${c.toLowerCase()}\\b`, "i").test(lower)) {
        city = c;
        break;
      }
    }
  }

  // Extract topics / technologies
  const topics = KNOWN_TOPICS.filter((t) => t.patterns.some((p) => p.test(query))).map((t) => t.label);

  // Extract audience
  const audience: string[] = [];
  if (/\bdevelopers?|engineers?\b/i.test(lower)) audience.push("developers");
  if (/\bdesigners?\b/i.test(lower)) audience.push("designers");
  if (/\bstudents?|juniors?\b/i.test(lower)) audience.push("students");
  if (/\bfounders?|leaders?|architects?\b/i.test(lower)) audience.push("architects");
  if (/\bdata scientists?|analysts?\b/i.test(lower)) audience.push("data scientists");

  // Extract date range
  const { dateStart, dateEnd } = extractDateRange(query, now);

  const isFreeOnly = /\bfree\b|\bno cost\b|\b\$0\b/.test(lower);

  const newFilters = sanitizeFilters({
    formats,
    categories,
    technologies: topics,
    dateStart,
    dateEnd,
    city,
  });

  const hasNewConstraints =
    formats.length > 0 ||
    categories.length > 0 ||
    topics.length > 0 ||
    Boolean(dateStart || dateEnd || city);

  const explicitFollowUpRef =
    Boolean(prevContext?.resultIds.length) &&
    (/\b(these|those|them|previous|above|current results?|same events?|from these|filter those|the ones?|ones|of them|narrow (?:it|them|down)|what about|how about)\b/i.test(
      lower,
    ) ||
      /^(?:and |but |ok(?:ay)?,? )?(?:only|just)\b/i.test(lower) ||
      /\bonly (?:the )?(?:online|in[- ]person|hybrid|free|virtual)\b/i.test(lower) ||
      /^(now\s+)?(compare|chart|graph|plot|sort|hide|show|group|display|make a|create a|turn (?:it|them|this)|build a dashboard|export|timeline|kanban|checklist)/i.test(
        lower,
      ));

  const tableCommand = extractTableCommand(query);
  const requestedChartType = detectChartType(lower);
  const requestedDimension = detectChartDimension(lower);
  const requestedDiagramType = detectDiagramType(lower);

  // Classify intent
  let intent: WorkspaceIntent = "search_events";
  if (
    /\b(diagram|flowchart|flow chart|architecture|sequence diagram|er diagram|entity relationship|mind\s*map|decision tree|user journey|state diagram|how semantic search works|how indexing works|how .* works)\b/i.test(
      lower,
    )
  ) {
    intent = "generate_diagram";
  } else if (/\b(create|build|generate|make|show)\b.{0,24}\b(form|filter form|feedback form|search form)\b/i.test(lower)) {
    intent = "generate_form";
  } else if (/\bdashboard\b|\banalytics overview\b|\bvisual overview\b/i.test(lower)) {
    intent = "build_dashboard";
  } else if (tableCommand && !/\bsearch\b|\bfind\b|\blook for\b/i.test(lower)) {
    intent = "refine_table";
  } else if (
    /\b(chart|graph|plot|histogram|distribution|visuali[sz]e|pie|donut|bar|line|scatter)\b/i.test(lower) &&
    !/\bcompare\b.{0,25}\btable\b/i.test(lower)
  ) {
    intent = "analytics_chart";
  } else if (/\bcompare\b|\bcomparison\b|\bside[- ]by[- ]side\b|\bvs\.?\b/i.test(lower)) {
    intent = "compare_events";
  } else if (/\btimeline\b|\bschedule\b|\bchronology\b/i.test(lower)) {
    intent = "timeline_view";
  } else if (/\bkanban\b|\bboard\b/i.test(lower)) {
    intent = "kanban_view";
  } else if (/\bchecklist\b|\baction items\b|\bprep list\b/i.test(lower)) {
    intent = "checklist_view";
  } else if (/\breport\b|\bsummari[sz]e\b|\bexecutive summary\b|\bexport\b/i.test(lower)) {
    intent = "export_report";
  } else if (query.length < 3) {
    intent = "clarify";
  }

  // Decide whether to reuse previous search result set vs run fresh search
  const reusePreviousResults =
    Boolean(prevContext?.resultIds.length) &&
    explicitFollowUpRef &&
    !topics.length &&
    !categories.length &&
    !dateStart &&
    !dateEnd &&
    !city &&
    (intent === "compare_events" ||
      intent === "analytics_chart" ||
      intent === "build_dashboard" ||
      intent === "refine_table" ||
      intent === "timeline_view" ||
      intent === "kanban_view" ||
      intent === "checklist_view" ||
      intent === "export_report" ||
      formats.length > 0);

  const clearPreviousFilters = /\b(reset|clear|all events|new search|start over)\b/i.test(lower);

  const effectiveFilters =
    explicitFollowUpRef && prevContext && !clearPreviousFilters
      ? mergeFilters(prevContext.filters, newFilters)
      : newFilters;

  const effectiveTopics =
    topics.length > 0
      ? topics
      : explicitFollowUpRef && prevContext?.filters.technologies.length
        ? prevContext.filters.technologies
        : [];

  // Search text: if follow-up filter refinement ("show only online events"), inherit previous searchText
  let searchText = query;
  if (explicitFollowUpRef && prevContext?.searchText && !topics.length && !hasNewConstraints) {
    searchText = prevContext.searchText;
  } else if (explicitFollowUpRef && prevContext?.searchText && formats.length > 0 && !topics.length) {
    searchText = prevContext.searchText;
  }

  const requestedTable = /\btable\b|\bcolumns?\b|\blist\b|\bcompare\b/i.test(lower);
  const requestedCards = /\bcards?\b|\bfind\b|\bdiscover\b|\bshow\b/i.test(lower);

  return {
    intent,
    searchText,
    topics: effectiveTopics,
    audience,
    filters: effectiveFilters,
    isFreeOnly,
    isFollowUp: explicitFollowUpRef,
    reusePreviousResults,
    clearPreviousFilters,
    ...(requestedChartType ? { requestedChartType } : {}),
    requestedDimension,
    requestedDiagramType,
    requestedTable,
    requestedCards,
    ...(tableCommand ? { tableCommand } : {}),
    confidence: topics.length > 0 || hasNewConstraints || intent !== "search_events" ? 0.92 : 0.72,
  };
}


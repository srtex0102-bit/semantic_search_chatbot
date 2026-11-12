import { describe, expect, it } from "vitest";
import { parseUserIntent } from "@/lib/ai/intent-parser";
import { EMPTY_FILTERS, type ConversationContext } from "@/lib/events/types";

const NOV_2026 = new Date("2026-11-10T12:00:00Z");

describe("natural language intent parser", () => {
  it("parses multi-constraint PRD showcase query with relative dates", () => {
    const parsed = parseUserIntent(
      "Find all online React and Next.js events next month, compare them by date and category, and show me an analytics dashboard.",
      null,
      NOV_2026,
    );
    expect(parsed.intent).toBe("build_dashboard");
    expect(parsed.filters.formats).toEqual(["online"]);
    expect(parsed.topics).toEqual(expect.arrayContaining(["React", "Next.js"]));
    expect(parsed.filters.dateStart).toBe("2026-12-01");
    expect(parsed.filters.dateEnd).toBe("2026-12-31");
  });

  it("handles conversational follow-up refinements without losing prior search context", () => {
    const prev: ConversationContext = {
      query: "Find React developer events",
      searchText: "Find React developer events",
      filters: { ...EMPTY_FILTERS, technologies: ["React"] },
      resultIds: ["11111111-1111-4111-8111-111111111111"],
      intent: "search_events",
      at: NOV_2026.toISOString(),
    };

    const onlyOnline = parseUserIntent("Show only online events", prev, NOV_2026);
    expect(onlyOnline.isFollowUp).toBe(true);
    expect(onlyOnline.reusePreviousResults).toBe(true);
    expect(onlyOnline.filters.formats).toEqual(["online"]);
    expect(onlyOnline.filters.technologies).toEqual(["React"]);

    const compare = parseUserIntent("Now compare them by date", prev, NOV_2026);
    expect(compare.intent).toBe("compare_events");
    expect(compare.reusePreviousResults).toBe(true);

    const chart = parseUserIntent("Create a donut chart by category", prev, NOV_2026);
    expect(chart.intent).toBe("analytics_chart");
    expect(chart.requestedChartType).toBe("donut");
    expect(chart.requestedDimension).toBe("category");
    expect(chart.reusePreviousResults).toBe(true);

    const refine = parseUserIntent("Hide the description column and sort by event date", prev, NOV_2026);
    expect(refine.intent).toBe("refine_table");
    expect(refine.tableCommand?.hideColumns).toContain("description");
    expect(refine.tableCommand?.sortBy).toBe("startAt");
  });

  it("detects diagrams and dynamic forms", () => {
    const flow = parseUserIntent("Show how semantic search works");
    expect(flow.intent).toBe("generate_diagram");
    expect(flow.requestedDiagramType).toBe("flowchart");

    const arch = parseUserIntent("Draw the complete architecture of this semantic search application");
    expect(arch.intent).toBe("generate_diagram");
    expect(arch.requestedDiagramType).toBe("architecture");

    const form = parseUserIntent("Create a form to filter events");
    expect(form.intent).toBe("generate_form");
  });

  it("keeps earlier technology and date filters when the user says 'only the online ones'", () => {
    const prev: ConversationContext = {
      query: "Find React events in December",
      searchText: "Find React events in December",
      filters: {
        ...EMPTY_FILTERS,
        technologies: ["React"],
        dateStart: "2026-12-01",
        dateEnd: "2026-12-31",
      },
      resultIds: ["11111111-1111-4111-8111-111111111111"],
      intent: "search_events",
      at: NOV_2026.toISOString(),
    };
    const parsed = parseUserIntent("only the online ones", prev, NOV_2026);
    expect(parsed.isFollowUp).toBe(true);
    expect(parsed.filters.formats).toEqual(["online"]);
    expect(parsed.filters.technologies).toEqual(["React"]);
    expect(parsed.filters.dateStart).toBe("2026-12-01");
    expect(parsed.filters.dateEnd).toBe("2026-12-31");
  });

  it("starts a fresh search for an unrelated new query", () => {
    const prev: ConversationContext = {
      query: "Find React events",
      searchText: "Find React events",
      filters: { ...EMPTY_FILTERS, technologies: ["React"] },
      resultIds: ["11111111-1111-4111-8111-111111111111"],
      intent: "search_events",
      at: NOV_2026.toISOString(),
    };
    const parsed = parseUserIntent("Find Python workshops in January", prev, NOV_2026);
    expect(parsed.isFollowUp).toBe(false);
    expect(parsed.filters.technologies).not.toContain("React");
  });
});

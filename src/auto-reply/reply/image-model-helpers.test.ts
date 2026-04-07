import { describe, expect, it, vi, beforeEach } from "vitest";
import type { OpenClawConfig } from "../../config/config.js";
import {
  collectImageModelKeys,
  isImageModel,
  prepareImageModelFallbacks,
  resolveChannelModelSupportsVision,
  resolveModelSupportsVision,
} from "./image-model-helpers.js";

// Mock modules at the top level
const mockBuildAllowedModelSet = vi.fn(() => ({ allowAny: true, allowedKeys: new Set() }));
const emptyAliasIndex = () => ({ byAlias: new Map(), byKey: new Map() });

vi.mock("../../agents/model-catalog.js", () => ({
  loadModelCatalog: vi.fn(async () => [
    {
      provider: "anthropic",
      id: "claude-opus-4-6",
      name: "Claude Opus 4.6",
      input: ["text", "image"],
    },
    { provider: "openai", id: "gpt-4o", name: "GPT-4o", input: ["text", "image"] },
    { provider: "openai", id: "gpt-4o-mini", name: "GPT-4o mini", input: ["text", "image"] },
    { provider: "anthropic", id: "claude-3-haiku", name: "Claude 3 Haiku", input: ["text"] },
  ]),
  findModelInCatalog: vi.fn((catalog, provider, model) =>
    catalog.find(
      (e: { provider: string; id: string }) => e.provider === provider && e.id === model,
    ),
  ),
  modelSupportsVision: vi.fn((entry) => entry?.input?.includes("image") ?? false),
}));

vi.mock("../../agents/model-selection.js", () => ({
  buildModelAliasIndex: vi.fn(() => emptyAliasIndex()),
  buildAllowedModelSet: () => mockBuildAllowedModelSet(),
  modelKey: vi.fn((provider, model) => `${provider}/${model}`),
  resolveModelRefFromString: vi.fn((params: { raw: string; defaultProvider?: string }) => {
    const raw = params.raw.trim();
    if (raw.includes("/")) {
      const [provider, ...modelParts] = raw.split("/");
      return {
        ref: { provider, model: modelParts.join("/") },
        alias: false,
      };
    }
    return {
      ref: { provider: params.defaultProvider ?? "unknown", model: raw },
      alias: false,
    };
  }),
}));

beforeEach(() => {
  mockBuildAllowedModelSet.mockReset();
  mockBuildAllowedModelSet.mockReturnValue({ allowAny: true, allowedKeys: new Set() });
});

describe("collectImageModelKeys", () => {
  it("returns empty set when no config", () => {
    const result = collectImageModelKeys({
      imageModelConfig: undefined,
      aliasIndex: emptyAliasIndex(),
      defaultProvider: "anthropic",
    });
    expect(result.keys.size).toBe(0);
    expect(result.imageModelDefaultProvider).toBe("anthropic");
  });

  it("collects string config", () => {
    const result = collectImageModelKeys({
      imageModelConfig: "openai/gpt-4o",
      aliasIndex: emptyAliasIndex(),
      defaultProvider: "anthropic",
    });
    expect(result.keys.has("openai/gpt-4o")).toBe(true);
    expect(result.imageModelDefaultProvider).toBe("openai");
  });

  it("collects primary with fallbacks", () => {
    const result = collectImageModelKeys({
      imageModelConfig: {
        primary: "anthropic/claude-opus-4-6",
        fallbacks: ["openai/gpt-4o", "openai/gpt-4o-mini"],
      },
      aliasIndex: emptyAliasIndex(),
      defaultProvider: "anthropic",
    });
    expect(result.keys.has("anthropic/claude-opus-4-6")).toBe(true);
    expect(result.keys.has("openai/gpt-4o")).toBe(true);
    expect(result.keys.has("openai/gpt-4o-mini")).toBe(true);
    expect(result.imageModelDefaultProvider).toBe("anthropic");
  });

  it("handles fallback-only config", () => {
    const result = collectImageModelKeys({
      imageModelConfig: {
        fallbacks: ["openai/gpt-4o", "anthropic/claude-opus-4-6"],
      },
      aliasIndex: emptyAliasIndex(),
      defaultProvider: "anthropic",
    });
    expect(result.keys.has("openai/gpt-4o")).toBe(true);
    expect(result.keys.has("anthropic/claude-opus-4-6")).toBe(true);
    expect(result.imageModelDefaultProvider).toBe("openai");
  });

  it("derives provider from first fallback with explicit provider", () => {
    const result = collectImageModelKeys({
      imageModelConfig: {
        primary: "gpt-4o", // providerless
        fallbacks: ["anthropic/claude-opus-4-6"],
      },
      aliasIndex: emptyAliasIndex(),
      defaultProvider: "openai",
    });
    expect(result.keys.has("gpt-4o")).toBe(true);
    expect(result.imageModelDefaultProvider).toBe("anthropic");
  });
});

describe("isImageModel", () => {
  it("matches exact provider/model key", () => {
    const keys = new Set(["openai/gpt-4o", "anthropic/claude-opus-4-6"]);
    expect(isImageModel("openai", "gpt-4o", keys)).toBe(true);
    expect(isImageModel("anthropic", "claude-opus-4-6", keys)).toBe(true);
    expect(isImageModel("openai", "gpt-4o-mini", keys)).toBe(false);
  });

  it("matches providerless keys", () => {
    const keys = new Set(["gpt-4o", "claude-opus-4-6"]);
    // Providerless entries match any provider with same model name
    expect(isImageModel("openai", "gpt-4o", keys)).toBe(true);
    expect(isImageModel("anthropic", "gpt-4o", keys)).toBe(true);
    expect(isImageModel("openai", "gpt-4o-mini", keys)).toBe(false);
  });

  it("matches mixed keys - providerless matches any provider", () => {
    const keys = new Set(["openai/gpt-4o", "claude-opus-4-6"]);
    expect(isImageModel("openai", "gpt-4o", keys)).toBe(true);
    // providerless claude-opus-4-6 matches any provider
    expect(isImageModel("anthropic", "claude-opus-4-6", keys)).toBe(true);
    expect(isImageModel("openai", "claude-opus-4-6", keys)).toBe(true);
    // not in keys at all
    expect(isImageModel("anthropic", "gpt-4o-mini", keys)).toBe(false);
  });
});

describe("prepareImageModelFallbacks", () => {
  it("returns empty array for no fallbacks", () => {
    const result = prepareImageModelFallbacks({
      fallbacks: [],
      cfg: {} as OpenClawConfig,
      aliasIndex: emptyAliasIndex(),
      defaultProvider: "anthropic",
    });
    expect(result).toEqual([]);
  });

  it("filters out empty strings", () => {
    const result = prepareImageModelFallbacks({
      fallbacks: ["openai/gpt-4o", "", "  ", "anthropic/claude-opus-4-6"],
      cfg: {} as OpenClawConfig,
      aliasIndex: emptyAliasIndex(),
      defaultProvider: "anthropic",
    });
    expect(result).toContain("openai/gpt-4o");
    expect(result).toContain("anthropic/claude-opus-4-6");
    expect(result.length).toBe(2);
  });

  it("canonicalizes provider-qualified keys", () => {
    const result = prepareImageModelFallbacks({
      fallbacks: ["openai/gpt-4o", "anthropic/claude-opus-4-6"],
      cfg: {} as OpenClawConfig,
      aliasIndex: emptyAliasIndex(),
      defaultProvider: "anthropic",
    });
    expect(result).toContain("openai/gpt-4o");
    expect(result).toContain("anthropic/claude-opus-4-6");
  });

  it("uses imageModelProvider for resolution when provided", () => {
    // When imageModelProvider differs from defaultProvider,
    // providerless fallbacks should resolve against imageModelProvider
    const result = prepareImageModelFallbacks({
      fallbacks: ["gpt-4o"], // providerless
      cfg: {} as OpenClawConfig,
      aliasIndex: emptyAliasIndex(),
      defaultProvider: "anthropic",
      imageModelProvider: "openai",
    });
    // Should resolve to openai/gpt-4o
    expect(result).toContain("openai/gpt-4o");
  });
});

describe("resolveChannelModelSupportsVision", () => {
  it("returns false when no channel model override", async () => {
    const result = await resolveChannelModelSupportsVision({
      channelModelOverride: undefined,
      imageModelConfig: { primary: "openai/gpt-4o" },
      defaultProvider: "anthropic",
      cfg: {} as OpenClawConfig,
      hasAppliedImageModelOverride: true,
    });
    expect(result.channelModelIsVisionModel).toBe(false);
  });

  it("returns false when hasAppliedImageModelOverride is false", async () => {
    const result = await resolveChannelModelSupportsVision({
      channelModelOverride: { model: "openai/gpt-4o" },
      imageModelConfig: { primary: "openai/gpt-4o" },
      defaultProvider: "anthropic",
      cfg: {} as OpenClawConfig,
      hasAppliedImageModelOverride: false,
    });
    expect(result.channelModelIsVisionModel).toBe(false);
  });

  it("returns true when channel model matches imageModel", async () => {
    const result = await resolveChannelModelSupportsVision({
      channelModelOverride: { model: "openai/gpt-4o" },
      imageModelConfig: { primary: "openai/gpt-4o" },
      defaultProvider: "anthropic",
      cfg: {} as OpenClawConfig,
      hasAppliedImageModelOverride: true,
    });
    expect(result.channelModelIsVisionModel).toBe(true);
    expect(result.channelResolved).toEqual({ provider: "openai", model: "gpt-4o" });
  });

  it("returns true when channel model matches imageModel fallback", async () => {
    const result = await resolveChannelModelSupportsVision({
      channelModelOverride: { model: "openai/gpt-4o-mini" },
      imageModelConfig: {
        primary: "anthropic/claude-opus-4-6",
        fallbacks: ["openai/gpt-4o", "openai/gpt-4o-mini"],
      },
      defaultProvider: "anthropic",
      cfg: {} as OpenClawConfig,
      hasAppliedImageModelOverride: true,
    });
    expect(result.channelModelIsVisionModel).toBe(true);
  });

  it("returns true when catalog indicates vision support", async () => {
    const result = await resolveChannelModelSupportsVision({
      channelModelOverride: { model: "anthropic/claude-opus-4-6" },
      imageModelConfig: { primary: "openai/gpt-4o" },
      defaultProvider: "anthropic",
      cfg: {} as OpenClawConfig,
      hasAppliedImageModelOverride: true,
    });
    // claude-opus-4-6 is in mocked catalog with vision support
    expect(result.channelModelIsVisionModel).toBe(true);
  });

  it("returns false for non-vision model", async () => {
    const result = await resolveChannelModelSupportsVision({
      channelModelOverride: { model: "anthropic/claude-3-haiku" },
      imageModelConfig: { primary: "openai/gpt-4o" },
      defaultProvider: "anthropic",
      cfg: {} as OpenClawConfig,
      hasAppliedImageModelOverride: true,
    });
    // claude-3-haiku in mock only supports text
    expect(result.channelModelIsVisionModel).toBe(false);
  });
});

describe("resolveModelSupportsVision", () => {
  it("returns true when model matches configured image model", async () => {
    await expect(
      resolveModelSupportsVision({
        provider: "openai",
        model: "gpt-4o",
        imageModelConfig: { primary: "openai/gpt-4o" },
        defaultProvider: "anthropic",
        cfg: {} as OpenClawConfig,
      }),
    ).resolves.toBe(true);
  });

  it("returns false for catalog text-only model", async () => {
    await expect(
      resolveModelSupportsVision({
        provider: "anthropic",
        model: "claude-3-haiku",
        imageModelConfig: { primary: "openai/gpt-4o" },
        defaultProvider: "anthropic",
        cfg: {} as OpenClawConfig,
      }),
    ).resolves.toBe(false);
  });
});

describe("image model auto-switch scenarios", () => {
  /**
   * Scenario 1: Dashboard sends image with allowlist configured
   * - imageModel primary is not in allowlist
   * - Should try fallbacks
   */
  describe("allowlist filtering", () => {
    it("filters fallbacks against allowlist", () => {
      // Mock buildAllowedModelSet to return a restricted allowlist
      mockBuildAllowedModelSet.mockReturnValueOnce({
        allowAny: false,
        allowedKeys: new Set(["openai/gpt-4o", "anthropic/claude-opus-4-6"]),
      });

      const result = prepareImageModelFallbacks({
        fallbacks: ["openai/gpt-4o", "openai/gpt-4o-mini", "anthropic/claude-3-haiku"],
        cfg: {} as OpenClawConfig,
        aliasIndex: emptyAliasIndex(),
        defaultProvider: "anthropic",
      });

      // Only gpt-4o is in allowlist
      expect(result).toContain("openai/gpt-4o");
      expect(result).not.toContain("openai/gpt-4o-mini");
      expect(result).not.toContain("anthropic/claude-3-haiku");
    });
  });

  /**
   * Scenario 2: Cross-provider image model switch
   * - Default provider is anthropic
   * - imageModel is openai/gpt-4o
   * - Need to ensure fallbacks resolve with correct provider context
   */
  describe("cross-provider resolution", () => {
    it("uses imageModelProvider for providerless fallbacks", () => {
      // When primary is "openai/gpt-4o" and fallbacks include providerless models
      const result = prepareImageModelFallbacks({
        fallbacks: ["gpt-4o-mini"], // providerless, should resolve as openai/gpt-4o-mini
        cfg: {} as OpenClawConfig,
        aliasIndex: emptyAliasIndex(),
        defaultProvider: "anthropic", // agent default is different from image model provider
        imageModelProvider: "openai",
      });

      // Should be resolved as openai/gpt-4o-mini, not anthropic/gpt-4o-mini
      expect(result).toContain("openai/gpt-4o-mini");
    });
  });

  /**
   * Scenario 3: Fallback-only imageModel config
   * - No primary configured
   * - First fallback becomes primary
   * - Provider derived from first fallback
   */
  describe("fallback-only config", () => {
    it("derives provider from first fallback when no primary", () => {
      const result = collectImageModelKeys({
        imageModelConfig: {
          fallbacks: ["openai/gpt-4o", "anthropic/claude-opus-4-6"],
        },
        aliasIndex: emptyAliasIndex(),
        defaultProvider: "anthropic",
      });

      // Provider should come from first fallback
      expect(result.imageModelDefaultProvider).toBe("openai");
      expect(result.keys.has("openai/gpt-4o")).toBe(true);
      expect(result.keys.has("anthropic/claude-opus-4-6")).toBe(true);
    });
  });

  /**
   * Scenario 4: Providerless fallbacks with cross-provider primary
   * - Primary is openai/gpt-4o
   * - Fallbacks include providerless models
   * - Providerless fallbacks should resolve against openai, not agent default
   */
  describe("providerless fallbacks with cross-provider primary", () => {
    it("resolves providerless fallbacks using imageModelProvider", () => {
      // This simulates the fix in chat.ts where imageModelProvider is derived
      // from modelOverride and passed to prepareImageModelFallbacks
      const result = prepareImageModelFallbacks({
        fallbacks: ["gpt-4o-mini", "gpt-4o"], // providerless
        cfg: {} as OpenClawConfig,
        aliasIndex: emptyAliasIndex(),
        defaultProvider: "anthropic", // agent default is different
        imageModelProvider: "openai", // derived from modelOverride
      });

      // Should be resolved as openai models, not anthropic
      expect(result).toContain("openai/gpt-4o-mini");
      expect(result).toContain("openai/gpt-4o");
    });
  });
});

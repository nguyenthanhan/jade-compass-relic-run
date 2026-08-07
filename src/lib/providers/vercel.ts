import { createOpenAI, OpenAIProvider } from "@ai-sdk/openai";
import { createAnthropic, AnthropicProvider } from "@ai-sdk/anthropic";
import {
  createGoogleGenerativeAI,
  GoogleGenerativeAIProvider,
} from "@ai-sdk/google";
import { createGroq, GroqProvider } from "@ai-sdk/groq";
import { createMistral, MistralProvider } from "@ai-sdk/mistral";
import {
  createOpenRouter,
  LanguageModelV4,
  OpenRouterProvider,
} from "@openrouter/ai-sdk-provider";
import { generateObject, generateText } from "ai";
import { ContentLanguageType, AISDKProviderType } from "@/types/llm";
import { IFullStoryResponse } from "@/games/relic-expedition/types";
import { BaseLLMProvider } from "./base";
import {
  FullStoryObjectSchema,
  validateFullStoryResponse,
} from "@/games/relic-expedition/lib/schemas/full-story";
import { parseJSONResponse } from "@/utils/string";
import { logger } from "../logger";

// Schema imported from shared module for generateObject
export class VercelAIProvider extends BaseLLMProvider {
  name: string;
  protected apiKey: string;
  protected apiBase: string;
  protected model: string;
  private aiProvider: AISDKProviderType;

  private openai?: OpenAIProvider;
  private anthropic?: AnthropicProvider;
  private google?: GoogleGenerativeAIProvider;
  private groq?: GroqProvider;
  private mistral?: MistralProvider;
  private openrouter?: OpenRouterProvider;

  constructor(
    apiKey: string,
    name: string,
    apiBase: string,
    model: string,
    aiProvider: AISDKProviderType = "openai-ai-sdk",
  ) {
    super(apiKey, apiBase, model);
    this.name = name;
    this.apiKey = apiKey;
    this.apiBase = apiBase;
    this.model = model;
    this.aiProvider = aiProvider;

    if (!apiKey) {
      throw new Error(`${name} API key is required`);
    }
  }

  private getAIModel(): LanguageModelV4 {
    switch (this.aiProvider) {
      case "openai-ai-sdk":
        if (!this.openai) {
          this.openai = createOpenAI({ apiKey: this.apiKey });
        }
        return this.openai(this.model);
      case "anthropic-ai-sdk":
        if (!this.anthropic) {
          this.anthropic = createAnthropic({ apiKey: this.apiKey });
        }
        return this.anthropic(this.model);
      case "google-ai-sdk":
        if (!this.google) {
          this.google = createGoogleGenerativeAI({ apiKey: this.apiKey });
        }
        return this.google(this.model);
      case "groq-ai-sdk":
        if (!this.groq) {
          this.groq = createGroq({ apiKey: this.apiKey });
        }
        return this.groq(this.model);
      case "mistral-ai-sdk":
        if (!this.mistral) {
          this.mistral = createMistral({ apiKey: this.apiKey });
        }
        return this.mistral(this.model);
      case "openrouter-ai-sdk":
        if (!this.openrouter) {
          this.openrouter = createOpenRouter({ apiKey: this.apiKey });
        }
        return this.openrouter(this.model);
      default:
        throw new Error(`Unsupported AI provider: ${this.aiProvider}`);
    }
  }

  async generateFullStoryByRawText(
    totalRounds: number,
    choicesPerRound: number,
    contentLanguage: ContentLanguageType,
    seed: string,
  ): Promise<IFullStoryResponse> {
    const requestId = this.generateRequestId();
    const systemPrompt = this.createFullStorySystemPrompt({
      contentLanguage,
    });
    const prompt = this.createFullStoryPrompt(
      totalRounds,
      choicesPerRound,
      true,
    );

    this.logRequest("generateFullStory", requestId, prompt, systemPrompt, {
      totalRounds,
      choicesPerRound,
      seed,
      model: this.model,
    });

    const startTime = Date.now();
    try {
      if (!this.apiKey) {
        throw new Error(`${this.name} API key is not configured`);
      }

      const response = await generateText({
        model: this.getAIModel(),
        system: systemPrompt,
        prompt: prompt,
        temperature: 0.7,
        seed: seed && !isNaN(parseInt(seed)) ? parseInt(seed) : undefined,
      });

      const responseTime = Date.now() - startTime;

      const jsonText = parseJSONResponse<object>(response.text);
      const parsedResponse = validateFullStoryResponse(jsonText, {
        rounds: totalRounds,
        choicesPerRound,
      });

      this.logResponse(
        requestId,
        response,
        jsonText,
        parsedResponse,
        responseTime,
        undefined,
        undefined,
      );

      return parsedResponse;
    } catch (error) {
      const responseTime = Date.now() - startTime;
      const errorMessage =
        error instanceof Error ? error.message : "Unknown error";

      this.logResponse(
        requestId,
        undefined,
        undefined,
        undefined,
        responseTime,
        errorMessage,
        undefined,
      );

      throw new Error(`Failed to generate story: ${errorMessage}`);
    }
  }

  async generateFullStory(
    totalRounds: number,
    choicesPerRound: number,
    contentLanguage: ContentLanguageType,
    seed: string,
  ): Promise<IFullStoryResponse> {
    const requestId = this.generateRequestId();
    const systemPrompt = this.createFullStorySystemPrompt({
      contentLanguage,
    });

    // not need to use plus format because generateObject will return the full response
    const prompt = this.createFullStoryPrompt(
      totalRounds,
      choicesPerRound,
      false,
    );

    this.logRequest("generateFullStory", requestId, prompt, systemPrompt, {
      totalRounds,
      choicesPerRound,
      seed,
      model: this.model,
    });

    const startTime = Date.now();
    try {
      if (!this.apiKey) {
        throw new Error(`${this.name} API key is not configured`);
      }

      const response = await generateObject({
        model: this.getAIModel(),
        schema: FullStoryObjectSchema,
        system: systemPrompt,
        prompt: prompt,
        temperature: 0.7,
        seed: seed && !isNaN(parseInt(seed)) ? parseInt(seed) : undefined,
      });

      const responseTime = Date.now() - startTime;

      // parse for logs, not need to return
      const parsedResponse = validateFullStoryResponse(response.object, {
        rounds: totalRounds,
        choicesPerRound,
      });

      this.logResponse(
        requestId,
        response,
        response.object,
        parsedResponse,
        responseTime,
        undefined,
        undefined,
      );

      return parsedResponse;
    } catch (error) {
      const responseTime = Date.now() - startTime;
      const errorMessage =
        error instanceof Error ? error.message : "Unknown error";

      this.logResponse(
        requestId,
        undefined,
        undefined,
        undefined,
        responseTime,
        errorMessage,
        undefined,
      );

      throw new Error(`Failed to generate story: ${errorMessage}`);
    }
  }

  async testModelsAvailability() {}

  async testConnection() {
    try {
      const result = await generateText({
        model: this.getAIModel(),
        prompt: "only return: OK",
      });
      logger.debug("Connection test result:", result.text);
    } catch (error) {
      const errorMessage =
        error instanceof Error ? error.message : String(error);
      throw new Error(`${errorMessage}`);
    }
  }
}

/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { GoogleGenAI } from '@google/genai';
import { FileHandler } from './fileHandler.js';
import { MemoryHandler } from './memoryHandler.js';
import * as path from 'path';
import * as fs from 'fs';
import {
  ImageGenerationRequest,
  ImageGenerationResponse,
  AuthConfig,
  StorySequenceArgs,
} from './types.js';
import { exec } from 'child_process';
import { promisify } from 'util';

const execAsync = promisify(exec);

export class ImageGenerator {
  private ai: GoogleGenAI;
  private modelName: string;
  private artModel: string;
  private textModel: string;
  private static readonly DEFAULT_MODEL = 'gemini-2.5-flash-image';
  private static readonly DEFAULT_TEXT_MODEL = 'gemini-3-pro-image-preview';

  constructor(authConfig: AuthConfig) {
    this.ai = new GoogleGenAI({
      apiKey: authConfig.apiKey,
    });
    this.artModel = process.env.NANOBANANA_ART_MODEL || ImageGenerator.DEFAULT_MODEL;
    this.textModel = process.env.NANOBANANA_TEXT_MODEL || ImageGenerator.DEFAULT_TEXT_MODEL;
    this.modelName = process.env.NANOBANANA_MODEL || this.artModel;
    
    console.error(`DEBUG - Models initialized: Art=${this.artModel}, Text=${this.textModel}, Default=${this.modelName}`);
  }

  private getSafetySettings(): any[] {
    return [
      { category: 'HARM_CATEGORY_HARASSMENT', threshold: 'BLOCK_NONE' },
      { category: 'HARM_CATEGORY_HATE_SPEECH', threshold: 'BLOCK_NONE' },
      { category: 'HARM_CATEGORY_SEXUALLY_EXPLICIT', threshold: 'BLOCK_NONE' },
      { category: 'HARM_CATEGORY_DANGEROUS_CONTENT', threshold: 'BLOCK_NONE' },
    ];
  }

  private async openImagePreview(filePath: string): Promise<void> {
    try {
      const platform = process.platform;
      let command: string;

      switch (platform) {
        case 'darwin': // macOS
          command = `open "${filePath}"`;
          break;
        case 'win32': // Windows
          command = `start "" "${filePath}"`;
          break;
        default: // Linux and others
          command = `xdg-open "${filePath}"`;
          break;
      }

      await execAsync(command);
      console.error(`DEBUG - Opened preview for: ${filePath}`);
    } catch (error: unknown) {
      console.error(
        `DEBUG - Failed to open preview for ${filePath}:`,
        error instanceof Error ? error.message : String(error),
      );
      // Don't throw - preview failure shouldn't break image generation
    }
  }

  private shouldAutoPreview(request: ImageGenerationRequest): boolean {
    // If --no-preview is explicitly set, never preview
    if (request.noPreview) {
      return false;
    }

    // Only preview when --preview flag is explicitly set
    if (request.preview) {
      return true;
    }

    // No auto-preview - images only open when explicitly requested
    return false;
  }

  private async handlePreview(
    files: string[],
    request: ImageGenerationRequest,
  ): Promise<void> {
    const shouldPreview = this.shouldAutoPreview(request);

    if (!shouldPreview || !files.length) {
      if (files.length > 1 && request.noPreview) {
        console.error(
          `DEBUG - Auto-preview disabled for ${files.length} images (--no-preview specified)`,
        );
      }
      return;
    }

    console.error(
      `DEBUG - ${request.preview ? 'Explicit' : 'Auto'}-opening ${files.length} image(s) for preview`,
    );

    // Open all generated images
    const previewPromises = files.map((file) => this.openImagePreview(file));
    await Promise.all(previewPromises);
  }

  private async logToDisk(message: string): Promise<void> {
      try {
          const logDir = FileHandler.ensureOutputDirectory();
          const logFile = path.join(logDir, 'nanobanana-output.log');
          const timestamp = new Date().toISOString();
          await fs.promises.appendFile(logFile, `[${timestamp}] ${message}\n`, 'utf-8');
      } catch (error) {
          console.error('DEBUG - Failed to write to log file:', error);
      }
  }

  private async logGeneration(modelName: string, generatedFiles: string[], referenceInfo?: string): Promise<void> {
    try {
      let logEntry = `Model: ${modelName}, Generated Files: ${generatedFiles.join(', ')}`;
      if (referenceInfo) {
          logEntry += `, Reference: ${referenceInfo}`;
      }
      await this.logToDisk(logEntry);
      console.error(`DEBUG - Logged generation to file.`);
    } catch (error) {
      console.error('DEBUG - Failed to write to log file:', error);
    }
  }

  static validateAuthentication(): AuthConfig {
    const nanoGeminiKey = process.env.NANOBANANA_GEMINI_API_KEY;
    if (nanoGeminiKey) {
      console.error('✓ Found NANOBANANA_GEMINI_API_KEY environment variable');
      return { apiKey: nanoGeminiKey, keyType: 'GEMINI_API_KEY' };
    }

    const nanoGoogleKey = process.env.NANOBANANA_GOOGLE_API_KEY;
    if (nanoGoogleKey) {
      console.error('✓ Found NANOBANANA_GOOGLE_API_KEY environment variable');
      return { apiKey: nanoGoogleKey, keyType: 'GOOGLE_API_KEY' };
    }

    const geminiKey = process.env.GEMINI_API_KEY;
    if (geminiKey) {
      console.error(
        '✓ Found GEMINI_API_KEY environment variable (fallback)',
      );
      return { apiKey: geminiKey, keyType: 'GEMINI_API_KEY' };
    }

    const googleKey = process.env.GOOGLE_API_KEY;
    if (googleKey) {
      console.error(
        '✓ Found GOOGLE_API_KEY environment variable (fallback)',
      );
      return { apiKey: googleKey, keyType: 'GOOGLE_API_KEY' };
    }

    throw new Error(
      'ERROR: No valid API key found. Please set NANOBANANA_GEMINI_API_KEY, NANOBANANA_GOOGLE_API_KEY, GEMINI_API_KEY, or GOOGLE_API_KEY environment variable.\n' +
        'For more details on authentication, visit: https://github.com/google-gemini/gemini-cli/blob/main/docs/cli/authentication.md',
    );
  }

  private isValidBase64ImageData(data: string): boolean {
    // Check if data looks like base64 image data
    if (!data || data.length < 100) {
      return false; // Too short to be meaningful image data
    }

    // Check if it's valid base64 format
    const base64Regex = /^[A-Za-z0-9+/]*={0,2}$/;
    if (!base64Regex.test(data)) {
      return false; // Not valid base64
    }

    // Additional check: base64 image data is typically quite long
    if (data.length < 1000) {
      console.error(
        'DEBUG - Skipping short data that may not be image:',
        data.length,
        'characters',
      );
      return false;
    }

    return true;
  }

  private buildBatchPrompts(request: ImageGenerationRequest): string[] {
    const prompts: string[] = [];
    const basePrompt = request.prompt;

    // If no batch options, return original prompt
    if (!request.styles && !request.variations && !request.outputCount) {
      return [basePrompt];
    }

    // Handle styles
    if (request.styles && request.styles.length > 0) {
      for (const style of request.styles) {
        prompts.push(`${basePrompt}, ${style} style`);
      }
    }

    // Handle variations
    if (request.variations && request.variations.length > 0) {
      const basePrompts = prompts.length > 0 ? prompts : [basePrompt];
      const variationPrompts: string[] = [];

      for (const baseP of basePrompts) {
        for (const variation of request.variations) {
          switch (variation) {
            case 'lighting':
              variationPrompts.push(`${baseP}, dramatic lighting`);
              variationPrompts.push(`${baseP}, soft lighting`);
              break;
            case 'angle':
              variationPrompts.push(`${baseP}, from above`);
              variationPrompts.push(`${baseP}, close-up view`);
              break;
            case 'color-palette':
              variationPrompts.push(`${baseP}, warm color palette`);
              variationPrompts.push(`${baseP}, cool color palette`);
              break;
            case 'composition':
              variationPrompts.push(`${baseP}, centered composition`);
              variationPrompts.push(`${baseP}, rule of thirds composition`);
              break;
            case 'mood':
              variationPrompts.push(`${baseP}, cheerful mood`);
              variationPrompts.push(`${baseP}, dramatic mood`);
              break;
            case 'season':
              variationPrompts.push(`${baseP}, in spring`);
              variationPrompts.push(`${baseP}, in winter`);
              break;
            case 'time-of-day':
              variationPrompts.push(`${baseP}, at sunrise`);
              variationPrompts.push(`${baseP}, at sunset`);
              break;
          }
        }
      }
      if (variationPrompts.length > 0) {
        prompts.splice(0, prompts.length, ...variationPrompts);
      }
    }

    // If no styles/variations but outputCount > 1, create simple variations
    if (
      prompts.length === 0 &&
      request.outputCount &&
      request.outputCount > 1
    ) {
      for (let i = 0; i < request.outputCount; i++) {
        prompts.push(basePrompt);
      }
    }

    // Limit to outputCount if specified
    if (request.outputCount && prompts.length > request.outputCount) {
      prompts.splice(request.outputCount);
    }

    return prompts.length > 0 ? prompts : [basePrompt];
  }

  async generateTextToImage(
    request: ImageGenerationRequest,
  ): Promise<ImageGenerationResponse> {
    try {
      const outputPath = FileHandler.ensureOutputDirectory();
      const generatedFiles: string[] = [];
      const prompts = this.buildBatchPrompts(request);
      let firstError: string | null = null;

      console.error(`DEBUG - Generating ${prompts.length} image variation(s)`);

      for (let i = 0; i < prompts.length; i++) {
        const currentPrompt = prompts[i];
        console.error(
          `DEBUG - Generating variation ${i + 1}/${prompts.length}:`,
          currentPrompt,
        );

        try {
          // Make API call for each variation
          const response = await this.ai.models.generateContent({
            model: this.modelName,
            config: {
              responseModalities: request.includeText ? ['IMAGE', 'TEXT'] : ['IMAGE'],
              safetySettings: this.getSafetySettings(),
            } as any,
            contents: [
              {
                role: 'user',
                parts: [{ text: currentPrompt }],
              },
            ],
          });

          console.error('DEBUG - API Response structure for variation', i + 1);

          if (response.candidates && response.candidates[0]?.content?.parts) {
            // Process image parts in the response
            for (const part of response.candidates[0].content.parts) {
              let imageBase64: string | undefined;

              if (part.inlineData?.data) {
                imageBase64 = part.inlineData.data;
                console.error('DEBUG - Found image data in inlineData:', {
                  length: imageBase64?.length ?? 0,
                  mimeType: part.inlineData.mimeType,
                });
              } else if (part.text && this.isValidBase64ImageData(part.text)) {
                imageBase64 = part.text;
                console.error(
                  'DEBUG - Found image data in text field (fallback)',
                );
              }

              if (imageBase64) {
                const filename = FileHandler.generateFilename(
                  request.styles || request.variations
                    ? currentPrompt
                    : request.prompt,
                  request.fileFormat,
                  i,
                );
                const fullPath = await FileHandler.saveImageFromBase64(
                  imageBase64,
                  outputPath,
                  filename,
                );
                generatedFiles.push(fullPath);
                await this.logGeneration(this.modelName, [fullPath]);
                console.error('DEBUG - Image saved to:', fullPath);
                break; // Only process first valid image per variation
              }
            }
          }
        } catch (error: unknown) {
          const errorMessage = this.handleApiError(error);
          if (!firstError) {
            firstError = errorMessage;
          }
          console.error(
            `DEBUG - Error generating variation ${i + 1}:`,
            errorMessage,
          );

          // If auth-related, stop immediately
          if (errorMessage.toLowerCase().includes('authentication failed')) {
            return {
              success: false,
              message: 'Image generation failed',
              error: errorMessage,
            };
          }
        }
      }

      if (generatedFiles.length === 0) {
        return {
          success: false,
          message: 'Failed to generate any images',
          error: firstError || 'No image data found in API responses',
        };
      }

      // Handle preview if requested
      await this.handlePreview(generatedFiles, request);

      return {
        success: true,
        message: `Successfully generated ${generatedFiles.length} image variation(s)`,
        generatedFiles,
      };
    } catch (error: unknown) {
      console.error('DEBUG - Error in generateTextToImage:', error);
      return {
        success: false,
        message: 'Failed to generate image',
        error: this.handleApiError(error),
      };
    }
  }

  private handleApiError(error: unknown): string {
    // Ideal: Check for a specific error code or type from the SDK
    // Fallback: Check for revealing strings in the error message
    const errorMessage =
      error instanceof Error ? error.message : String(error).toLowerCase();

    if (errorMessage.includes('api key not valid')) {
      return 'Authentication failed: The provided API key is invalid. Please check your NANOBANANA_GEMINI_API_KEY environment variable.';
    }

    if (errorMessage.includes('permission denied')) {
      return 'Authentication failed: The provided API key does not have the necessary permissions for the Gemini API. Please check your Google Cloud project settings.';
    }

    if (errorMessage.includes('quota exceeded')) {
      return 'API quota exceeded. Please check your usage and limits in the Google Cloud console.';
    }

    // Check for GoogleGenerativeAIResponseError
    if (
      error &&
      typeof error === 'object' &&
      'response' in error &&
      error.response
    ) {
      const responseError = error as {
        response: { status: number; statusText: string };
      };
      const { status } = responseError.response;

      switch (status) {
        case 400:
          return 'The request was malformed. This may be due to an issue with the prompt. Please check for safety violations or unsupported content.';
        case 403: // General permission error if specific message not caught
          return 'Authentication failed. Please ensure your API key (e.g., NANOBANANA_GEMINI_API_KEY) is valid and has the necessary permissions.';
        case 500:
          return 'The image generation service encountered a temporary internal error. Please try again later.';
        default:
          return `API request failed with status ${status}. Please check your connection and API key.`;
      }
    }

    // Fallback for other error types
    return `An unexpected error occurred: ${errorMessage}`;
  }

    async generateStorySequence(
      request: ImageGenerationRequest,
      args?: StorySequenceArgs,
    ): Promise<ImageGenerationResponse> {
      try {
        const outputPath = FileHandler.ensureOutputDirectory();
        const generatedFiles: string[] = [];
        const steps = request.outputCount || 4;
        const type = args?.type || 'story';
        const style = args?.style || 'consistent';
        const transition = args?.transition || 'smooth';
        let firstError: string | null = null;
  
        console.error(`DEBUG - Generating ${steps}-step ${type} sequence`);
  
        // Generate each step of the story/process
        for (let i = 0; i < steps; i++) {
          const stepNumber = i + 1;
          let stepPrompt = `${request.prompt}, step ${stepNumber} of ${steps}`;
  
          // Add context based on type
          switch (type) {
            case 'story':
              stepPrompt += `, narrative sequence, ${style} art style`;
              break;
            case 'process':
              stepPrompt += `, procedural step, instructional illustration`;
              break;
            case 'tutorial':
              stepPrompt += `, tutorial step, educational diagram`;
              break;
            case 'timeline':
              stepPrompt += `, chronological progression, timeline visualization`;
              break;
          }
  
          // Add transition context
          if (i > 0) {
            stepPrompt += `, ${transition} transition from previous step`;
          }
  
          console.error(`DEBUG - Generating step ${stepNumber}: ${stepPrompt}`);
  
          try {
            const response = await this.ai.models.generateContent({
              model: this.modelName,
              config: {
                responseModalities: request.includeText ? ['IMAGE', 'TEXT'] : ['IMAGE'],
                safetySettings: this.getSafetySettings(),
              } as any,
              contents: [
                {
                  role: 'user',
                  parts: [{ text: stepPrompt }],
                },
              ],
            });
  
            if (response.candidates && response.candidates[0]?.content?.parts) {
              for (const part of response.candidates[0].content.parts) {
                let imageBase64: string | undefined;
  
                if (part.inlineData?.data) {
                  imageBase64 = part.inlineData.data;
                } else if (part.text && this.isValidBase64ImageData(part.text)) {
                  imageBase64 = part.text;
                }
  
                if (imageBase64) {
                  const filename = FileHandler.generateFilename(
                    `${type}step${stepNumber}${request.prompt}`,
                    'png', // Stories default to png
                    0,
                  );
                  const fullPath = await FileHandler.saveImageFromBase64(
                    imageBase64,
                    outputPath,
                    filename,
                  );
                  generatedFiles.push(fullPath);
                  await this.logGeneration(this.modelName, [fullPath]);
                  console.error(`DEBUG - Step ${stepNumber} saved to:`, fullPath);
                  break;
                }
              }
            }
          } catch (error: unknown) {
            const errorMessage = this.handleApiError(error);
            if (!firstError) {
              firstError = errorMessage;
            }
            console.error(
              `DEBUG - Error generating step ${stepNumber}:`,
              errorMessage,
            );
            if (errorMessage.toLowerCase().includes('authentication failed')) {
              return {
                success: false,
                message: 'Story generation failed',
                error: errorMessage,
              };
            }
          }
  
          // Check if this step was actually generated
          if (generatedFiles.length < stepNumber) {
            console.error(
              `DEBUG - WARNING: Step ${stepNumber} failed to generate - no valid image data received`,
            );
          }
        }
  
        console.error(
          `DEBUG - Story generation completed. Generated ${generatedFiles.length} out of ${steps} requested images`,
        );
  
        if (generatedFiles.length === 0) {
          return {
            success: false,
            message: 'Failed to generate any story sequence images',
            error: firstError || 'No image data found in API responses',
          };
        }
  
        // Handle preview if requested
        await this.handlePreview(generatedFiles, request);
  
        const wasFullySuccessful = generatedFiles.length === steps;
        const successMessage = wasFullySuccessful
          ? `Successfully generated complete ${steps}-step ${type} sequence`
          : `Generated ${generatedFiles.length} out of ${steps} requested ${type} steps (${steps - generatedFiles.length} steps failed)`;
  
        return {
          success: true,
          message: successMessage,
          generatedFiles,
        };
      } catch (error: unknown) {
        console.error('DEBUG - Error in generateStorySequence:', error);
        return {
          success: false,
          message: `Failed to generate ${request.mode} sequence`,
          error: this.handleApiError(error),
        };
      }
    }

  private extractImagePaths(text: string): string[] {
    // Looks for strings that look like file paths with image extensions
    // e.g. "characters/stan.png", "./ref/image.jpg"
    const regex = /(?:[\w\-\.\/\\\:]+)\.(?:png|jpg|jpeg|webp)/gi;
    const matches = text.match(regex) || [];
    return [...new Set(matches)]; // Return unique paths
  }

  private getAspectRatioInstruction(layout?: string): string {
    switch (layout) {
      case 'webtoon':
        return 'aspect ratio 9:16, vertical orientation';
      case 'strip':
        return 'aspect ratio 16:9, landscape orientation';
      case 'single_page':
        return 'aspect ratio 3:4, portrait orientation';
      case 'square':
      default:
        return 'aspect ratio 1:1, square format';
    }
  }

  private getAspectRatioString(layout?: string): string {
    switch (layout) {
      case 'webtoon':
        return '9:16';
      case 'strip':
        return '16:9';
      case 'single_page':
        return '3:4';
      case 'square':
      default:
        return '1:1';
    }
  }

  private delay(ms: number): Promise<void> {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  private async reviewGeneratedImage(
    generatedImagePath: string,
    references: { data: string; mimeType: string; sourcePath: string }[],
    options: {
        minScore?: number;
        minLikeness?: number;
        minStory?: number;
        minContinuity?: number;
        minLettering?: number;
        minNoBubbles?: number;
        storyContext?: string;
        isPhase1?: boolean;
        isColor?: boolean;
    } = { minScore: 8 }
  ): Promise<{ pass: boolean; score: number; reason: string; likeness_score?: number; lettering_score?: number }> {
    const minScore = options.minScore || 8;
    const { storyContext, minLikeness, minStory, minContinuity, isPhase1, isColor } = options;

    console.error(`DEBUG - Auto-Reviewing generated image for character consistency (Min Score: ${minScore})...`);
    
    // Filter references to prioritize characters (based on path/name)
    const characterRefs = references.filter(ref => 
        ref.sourcePath.includes('characters/') || 
        ref.sourcePath.includes('_portrait')
    );

    if (characterRefs.length === 0) {
        console.error('DEBUG - No character references found for review. Skipping.');
        return { pass: true, score: 10, reason: "No references to check against." };
    }

    try {
        const generatedB64 = await FileHandler.readImageAsBase64(generatedImagePath);
        
        const prompt = `You are a strict Quality Assurance AI for a manga production pipeline.
        Task: Compare the "Generated Image" with the provided "Reference Images" (including "Previous Page Reference" if available) AND the "Story Description".
        
        ${isPhase1 ? 'PHASE: ART PHASE (No Speech Bubbles allowed. Note: Captions, sound effects, and background text are PERMITTED.)' : 'PHASE: FINAL PHASE (Lettering/Color included)'}
        ${isColor ? 'TARGET FORMAT: FULL COLOR. (If the Story Description says "black and white", IGNORE IT. The user requested COLOR.)' : 'TARGET FORMAT: BLACK AND WHITE (Manga Style).'}

        STORY DESCRIPTION / CONTEXT:
        "${storyContext || 'No specific story text provided.'}"

        EVALUATION CRITERIA (Scored out of 100% each):
        1. [CRITICAL] Character Design & Identity (100% max): Does the character look EXACTLY like the main Character Reference sheet? Check eye shape, hair style/bangs, facial structure, BODY TYPE, and COSTUME. Identity and Design must be 100% consistent with the Ground Truth Character Sheet.
        2. [CRITICAL] Continuity (100% max): Does the overall visual style (line weight, shading, lighting) match the "Previous Page Reference"?
        3. [CRITICAL] ${isPhase1 ? 'NO SPEECH BUBBLES' : 'Lettering & Text'} (100% max): 
           ${isPhase1 ? 'Does the image contain any round SPEECH BUBBLES or THOUGHT BUBBLES? These are forbidden. NOTE: Rectangular caption boxes, sound effects (SFX), and incidental text on objects/walls are ALL PERMITTED. Only actual dialogue bubbles (usually white ovals with tails) are a failure.' : 'Are ALL speech bubbles and caption boxes filled with the CORRECT text from the Story Description? 1. Check for GIBBERISH. 2. Check for MISSING DIALOGUE. 3. Check for ALTERED TEXT. The text in the image must match the script WORD-FOR-WORD. Paraphrasing is a FAILURE.'}
        4. [CRITICAL] Story Accuracy & Panel Layout (100% max): Does the image match the provided Story Description (actions, emotions, items) AND the PANEL LAYOUT? If the script describes multiple panels (e.g., "Panel 1", "Panel 2"), the image MUST show that structure. If it describes a splash page, it must be one large image.

        TOTAL POSSIBLE SCORE: 400%.
        10/10 quality in all categories equals 400%.

        SCORING RUBRIC (Be Extremely Strict):
        - 100%: Perfect match. Identical face, hair, costume, and layout. ${isPhase1 ? 'No speech bubbles.' : 'All text matches script WORD-FOR-WORD (no missing lines, no typos).'}
        - 90%: Excellent likeness and layout. ${isPhase1 ? 'No speech bubbles.' : 'No empty bubbles, maybe 1 minor typo.'} Only pixel-level differences.
        - 70-80%: Recognizable, but minor costume or layout details are off. FACE MUST MATCH.
        - 50-60%: Looks like a different person, wrong outfit, wrong panel count, OR ${isPhase1 ? 'Contains speech bubbles' : 'contains GIBBERISH, MISSING DIALOGUE, or PARAPHRASED text'}.
        - 10-40%: Completely wrong person, wrong layout, or text is missing entirely.

        CRITICAL PENALTIES:
        ${isPhase1 ? '- [STRICT] SPEECH BUBBLES: If ANY round speech bubble or thought bubble is found, the no_bubbles_score MUST be below 40%. Captions, boxes, and SFX are allowed.' : '- [STRICT] TEXT ACCURACY: If ANY text is missing, gibberish, or paraphrased (different words than script), the lettering_score MUST be below 40%.\n        - [STRICT] NO DUPLICATES: If the same line of dialogue appears twice (e.g. once in a good bubble, once in a bad/ghost bubble), the lettering_score MUST be below 60%.'}
        ${isColor ? `- [CONDITIONAL] COLOR CONSISTENCY: 
           - EXCEPTION FOR PHASE 1: IGNORE ALL COLOR MISMATCHES. If the image is B&W, Sepia, or wrong colors, DO NOT PENALIZE. Assume Phase 2 will handle all colorization.
           - FOR PHASE 2: Compare hair/eye/costume colors. If the Character Reference is B&W, IGNORE color differences. If the Reference IS Color, strict consistency is required (likeness_score < 60% if mismatched).` : 
           '- [STRICT] COLOR: If the image is in Color despite "TARGET FORMAT: BLACK AND WHITE", the story_score MUST be below 50%.'}
        - [STRICT] PANEL LAYOUT: Count the panels. If the script asks for a 3-panel stack but the image is a single splash, the story_score MUST be below 50%.
        - If the visual style (shading/art style) clashes with the "Previous Page Reference" (IGNORING Color vs B&W differences in Phase 1), the continuity_score MUST be below 80%.
        - [STRICT] FACIAL IDENTITY: Compare the eyes, nose, and jawline. If it looks like a different person from the Character Reference, the likeness_score MUST be below 60%.
        - [STRICT] HAIR: The hairstyle (bangs, length, volume) must match the Main Reference exactly. If the hair is different, the likeness_score MUST be below 60%.
        - [STRICT] CLOTHING: The costume DESIGN must be consistent with the reference UNLESS the Story Description or Global Context explicitly describes a different outfit. If the BASE DESIGN changes without reason, the likeness_score MUST be below 60%.
        - If the image contradicts the Story Description (e.g. "fat" in text but "slim" in image), the story_score MUST be below 50%.
        
        STRICTLY enforce identity and ${isPhase1 ? 'ABSENCE of structural lettering elements (bubbles/captions)' : 'FULL TEXT completion'}. Do not allow "style" to excuse facial drift or ${isPhase1 ? 'bubbles' : 'empty bubbles'}.
        
        Output strictly in JSON format:
        {
            "likeness_score": number, // 0-100
            "continuity_score": number, // 0-100
            "${isPhase1 ? 'no_bubbles_score' : 'lettering_score'}": number, // 0-100
            "story_score": number, // 0-100
            "total_score": number, // 0-400
            "reason": "string", // Specific feedback on what is wrong.
            "pass": boolean // true if total_score >= 340 AND likeness_score >= 70 AND ${isPhase1 ? 'no_bubbles_score' : 'lettering_score'} >= 95
        }`;

        const parts: any[] = [{ text: prompt }];
        
        // Add Generated Image
        parts.push({ text: "Generated Image:" });
        parts.push({ inlineData: { data: generatedB64, mimeType: 'image/png' } });

        // Add References (Pass ALL references now, including environments)
        for (const ref of references) {
            const label = path.basename(ref.sourcePath);
            parts.push({ text: `Reference (${label}):` });
            parts.push({ inlineData: { data: ref.data, mimeType: ref.mimeType } });
        }

        const response = await this.ai.models.generateContent({
            model: this.textModel, 
            config: {
                responseModalities: ['TEXT'],
                responseMimeType: 'application/json',
                safetySettings: this.getSafetySettings(),
            } as any,
            contents: [{ role: 'user', parts: parts }],
        });

        const responseText = response.candidates?.[0]?.content?.parts?.[0]?.text;
        if (!responseText) {
            throw new Error("No response from review model");
        }

        // Clean up Markdown code blocks if present
        const cleanedText = responseText.replace(/```json\n?|\n?```/g, '').trim();
        let result;
        try {
            result = JSON.parse(cleanedText);
        } catch (parseError) {
            console.error(`DEBUG - JSON Parse Error: ${parseError}. Raw Text: ${responseText}`);
            return { pass: true, score: 0, reason: "Review failed to parse JSON response." };
        }

        // Ensure default values if fields are missing
        result.total_score = result.total_score ?? 0;
        result.likeness_score = result.likeness_score ?? 0;
        result.story_score = result.story_score ?? 0;
        result.continuity_score = result.continuity_score ?? 0;
        result.no_bubbles_score = result.no_bubbles_score ?? 0;
        result.lettering_score = result.lettering_score ?? 0;
        result.reason = result.reason || "No reason provided.";
        
        // Dynamic threshold based on minScore (1-10). Default 8 -> 320/400.
        const scoreThreshold = minScore * 40;

        // Specific thresholds
        const thresholdLikeness = minLikeness ? minLikeness * 10 : 70; // Default 70% safety
        const thresholdStory = minStory ? minStory * 10 : 70; // Default 70%
        const thresholdContinuity = minContinuity ? minContinuity * 10 : 70; // Default 70%
        const thresholdLettering = isPhase1 
            ? (options.minNoBubbles ? options.minNoBubbles * 10 : (options.minLettering ? options.minLettering * 10 : 50))
            : (options.minLettering ? options.minLettering * 10 : 95);

        // Enforce logic: Total score < threshold is a failure.
        const calculatedPass = 
            result.total_score >= scoreThreshold && 
            result.likeness_score >= thresholdLikeness &&
            result.story_score >= thresholdStory &&
            result.continuity_score >= thresholdContinuity &&
            (isPhase1 ? (result.no_bubbles_score >= thresholdLettering) : (result.lettering_score >= thresholdLettering));
        
        const phaseLabel = isPhase1 ? "Art" : "Final";
        const specialScoreLabel = isPhase1 ? "NoBubbles" : "Lettering";
        const specialScoreValue = isPhase1 ? result.no_bubbles_score : result.lettering_score;

        const logMsg = `[Auto-Review ${phaseLabel}] Model: ${this.textModel}. Total: ${result.total_score}/400% (Likeness: ${result.likeness_score}%, Continuity: ${result.continuity_score}%, Story: ${result.story_score}%, ${specialScoreLabel}: ${specialScoreValue}%). Threshold: ${scoreThreshold}% & Likeness >= ${thresholdLikeness}% & ${specialScoreLabel} >= ${thresholdLettering}%. Pass: ${calculatedPass}. Reason: ${result.reason}`;
        console.error(`DEBUG - ${logMsg}`);
        
        // Log to file
        try {
            const logDir = FileHandler.ensureOutputDirectory();
            const logFile = path.join(logDir, 'nanobanana-output.log');
            const timestamp = new Date().toISOString();
            await fs.promises.appendFile(logFile, `[${timestamp}] ${logMsg}\n`, 'utf-8');
        } catch (e) {
            console.error('DEBUG - Failed to log review to file:', e);
        }

        return {
            pass: calculatedPass,
            score: result.total_score,
            reason: result.reason,
            likeness_score: result.likeness_score,
            lettering_score: isPhase1 ? result.no_bubbles_score : result.lettering_score
        };

    } catch (error) {
        console.error('DEBUG - Auto-review failed (error calling model):', error);
        return { pass: true, score: 0, reason: "Review failed to execute." };
    }
  }

  private async addTextToMangaPage(
    imagePath: string,
    storyContent: string,
    pageHeader: string,
    references: { data: string; mimeType: string; sourcePath: string }[],
    isColor: boolean,
    artPrompt?: string,
    phase1Correction?: string,
    promptsDir?: string
  ): Promise<string> {
    console.error(`DEBUG - Phase 2: Adding text ${isColor ? 'and color ' : ''}to ${pageHeader} using ${this.textModel}...`);
    if (artPrompt) {
        const msg = `DEBUG - Phase 2 received visual context from Phase 1 prompt (${artPrompt.length} chars).`;
        console.error(msg);
        // We can't easily access logToDisk here without binding or passing it, but console.error should show up in CLI output.
        // Assuming user checks CLI output or we add a helper. 
        // Actually, I can use the same FileHandler to append if I really want to be safe, but console is good start.
    }
    
    try {
        const imageB64 = await FileHandler.readImageAsBase64(imagePath);
        
        // Extract Dialogue for Checklist to prevent missing text
        const dialogRegex = /(?:^|\n)\s*(?:[-*]\s*)?(?:\*\*)?([a-zA-Z0-9 '\-]+?)(?:\*\*)?\s*:\s*["“](.*?)["”]/g;
        const dialogueList: string[] = [];
        let match;
        while ((match = dialogRegex.exec(storyContent)) !== null) {
            dialogueList.push(`${match[1].trim()}: "${match[2].trim()}"`);
        }
        
        let checklist = "";
        if (dialogueList.length > 0) {
            checklist = "\n[MANDATORY DIALOGUE CHECKLIST]\nYou MUST include the following dialogue lines EXACTLY as written. Do not skip any:\n";
            dialogueList.forEach((line, index) => {
                checklist += `${index + 1}. ${line}\n`;
            });
            checklist += "Double-check that ALL lines above are present in the final image.\n";
        }

        let prompt = `You are a professional manga editor and artist. 
        Task: Add dialogue bubbles and text from the provided story script onto the attached manga page art. 
        ${isColor ? 'Also, COLORIZE the page using the provided reference images.' : ''}
        
        STORY SCRIPT FOR ${pageHeader}:
        "${storyContent}"
        ${checklist}
        
        ${artPrompt ? `\n[VISUAL CONTEXT FROM ART PHASE]\nThe attached "Generated Image" was created with this description. Use it to understand the intended colors, lighting, and atmosphere:\n"${artPrompt}"` : ''}
        
        ${phase1Correction ? `\n[CORRECTION INSTRUCTION]\nThe input image has a flaw: "${phase1Correction}".\nFix this by placing a correct dialogue bubble OVER the erroneous artifact (e.g. cover the empty/bad bubble) or ensuring the final composition hides it.` : ''}

        INSTRUCTIONS:
        1. **Create Dialogue Bubbles**: Analyze the script and the panels in the attached art. Create speech bubbles and caption boxes that fit the dialogue and composition.
        2. **Sequential Mapping**: Map the dialogue lines in the script to the bubbles you create in reading order.
        3. **Lettering**: Render ALL dialogue and captions into the bubbles/boxes. Use professional manga lettering style. Ensure text is centered and legible.
        4. **Verification**: EVERY line of dialogue and EVERY caption from the script MUST be present.
        5. **NO HALLUCINATIONS**: Do NOT add any random text, gibberish, or text not found in the script. All text in the image must come strictly from the provided story script. Check for typos.
        6. **CLEANUP & DEDUPLICATION**: The input art might contain "ghost" bubbles, faint text, or artifacts from the drawing phase. You MUST COVER or OVERPAINT these with your new, correct bubbles or artwork edits. Ensure there is NO DUPLICATE TEXT (e.g., the same line appearing twice). The final image must only contain the clean, sharp text from the script.`;

        if (isColor) {
            prompt += `
        5. **Colorization**: The attached page is in Black and White. You MUST colorize it.
           - Use the attached "Reference Image" portraits to match the EXACT hair, eye, skin, and costume colors for characters.
           - Use the "Far View" environment reference for background colors.
           - Ensure consistent color palettes across the entire page.`;
        } else {
            prompt += `
        5. **Maintain Style**: The attached page art is in Black and White. Keep the final output in professional Black and White manga style (screentones, ink, etc.). Do NOT add any color.`;
        }

        prompt += `
        6. **Art Integrity**: Maintain the original character likenesses and composition from the attached art. Do NOT redraw the panels, only overlay the bubbles, text, and color.
        7. Return the final high-quality image.`;
        
        // Save Phase 2 Prompt
        if (promptsDir) {
            try {
                const safeHeader = FileHandler.getSanitizedBaseName(pageHeader);
                const promptFile = path.join(promptsDir, `page_${safeHeader}_phase2.txt`);
                await FileHandler.saveTextFile(promptFile, prompt);
            } catch (e) { console.error('DEBUG - Failed to save Phase 2 prompt:', e); }
        }

        const parts: any[] = [
            { text: prompt },
            { inlineData: { data: imageB64, mimeType: 'image/png' } }
        ];

        // Add References for Colorization/Context
        for (const ref of references) {
            const label = path.basename(ref.sourcePath, path.extname(ref.sourcePath)).replace(/_/g, ' ');
            parts.push({ text: `Reference image for: "${label}"` });
            parts.push({ inlineData: { data: ref.data, mimeType: ref.mimeType } });
        }

        const response = await this.ai.models.generateContent({
            model: this.textModel,
            config: {
                responseModalities: ['IMAGE'],
                safetySettings: this.getSafetySettings(),
            } as any,
            contents: [{ 
                role: 'user', 
                parts: parts
            }],
        });

        if (response.candidates && response.candidates[0]?.content?.parts) {
            for (const part of response.candidates[0].content.parts) {
                let b64: string | undefined;
                if (part.inlineData?.data) b64 = part.inlineData.data;
                else if (part.text && this.isValidBase64ImageData(part.text)) b64 = part.text;

                if (b64) {
                    const dir = path.dirname(imagePath);
                    // Strip _phase_1 and any extension to get clean base
                    const cleanBase = path.basename(imagePath, '.png').replace(/_phase_1$/, '');
                    const newFileName = `${cleanBase}_final.png`;
                    const newPath = path.join(dir, newFileName);
                    await FileHandler.saveImageFromBase64(b64, dir, newFileName);
                    
                    await this.logGeneration(this.textModel, [newPath], `Phase 2 (Lettering) for ${pageHeader}`);
                    console.error(`DEBUG - Phase 2 SUCCESS: Saved final image to ${newPath}`);
                    return newPath;
                }
            }
        }
        throw new Error("No image data returned from text model");
    } catch (error) {
        console.error(`DEBUG - Phase 2 FAILED:`, error);
        throw error;
    }
  }

  async generateMangaPage(
    request: ImageGenerationRequest,
  ): Promise<ImageGenerationResponse> {
    try {
      // Handle Character Creation Mode
      if (request.createCharacter) {
         if (!request.inputImage || !request.storyFile) {
             return {
                 success: false,
                 message: 'Character creation requires both --image (source photo) and --file (story context/location).',
                 error: 'Missing inputImage or storyFile',
             };
         }

         console.error(`DEBUG - Character Creation Mode: Converting ${request.inputImage} for ${request.storyFile}`);
         
         const sourceFileRes = FileHandler.findInputFile(request.inputImage);
         if (!sourceFileRes.found) {
             return { success: false, message: `Source image not found: ${request.inputImage}` };
         }
         
         const storyFileRes = FileHandler.findInputFile(request.storyFile);
         // If story file doesn't exist, we can use current dir, but let's try to respect the path if provided as a location anchor
         const storyDir = storyFileRes.found ? path.dirname(storyFileRes.filePath!) : path.dirname(path.resolve(request.storyFile));
         const charsDir = path.join(storyDir, 'characters');
         FileHandler.ensureDirectory(charsDir);
         
         const promptsDir = path.join(storyDir, 'prompts');
         FileHandler.ensureDirectory(promptsDir);

         const sourceName = path.basename(request.inputImage, path.extname(request.inputImage));
         const safeName = sourceName.toLowerCase().replace(/[^a-z0-9]/g, '_');
         
         const bwFilename = `${safeName}_portrait.png`;
         const colorFilename = `${safeName}_portrait_color.png`;
         
         const sourceB64 = await FileHandler.readImageAsBase64(sourceFileRes.filePath!);
         const generatedFiles: string[] = [];
         let bwRefBase64: string | undefined;

         // Look for character description in story file
         let characterDescription = '';
         if (storyFileRes.found) {
             try {
                const storyText = await FileHandler.readTextFile(storyFileRes.filePath!);
                
                // Allow matching "the_martyr" against "The Martyr"
                // Escape special chars but replace underscores with space-or-underscore regex
                let namePattern = sourceName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
                namePattern = namePattern.replace(/_/g, '[\\s_]+');
                
                // 1. Try Header Style FIRST (Prioritize definitions over dialogue)
                // Matches: ### Name (Alias) \n Description
                const headerRegex = new RegExp(
                    `(?:^|\\n)#{1,6}\\s*(${namePattern})(?:[^\\n]*)?\\n+([^#\\n][\\s\\S]*?)(?=\\n#|$)`,
                    'i'
                );
                let match = storyText.match(headerRegex);
                
                if (match) {
                    characterDescription = match[2].trim();
                    console.error(`DEBUG - Found Header description for ${sourceName}: ${characterDescription.substring(0, 50)}...`);
                } else {
                    // 2. Try List Style: - **Name**: Description
                    const listRegex = new RegExp(`(?:^|\\n)\\s*[\\*\\-]?\\s*\\*?\\*?(${namePattern})\\*?\\*?(?::)?\\s*([^\\n]+)`, 'i');
                    match = storyText.match(listRegex);
                    
                    if (match) {
                        characterDescription = match[2].trim();
                        console.error(`DEBUG - Found List description for ${sourceName}: ${characterDescription}`);
                    }
                }
             } catch (e) {
                 console.error(`DEBUG - Failed to read/parse story file for description:`, e);
             }
         }

         // 1. Generate B&W Sheet
         const bwFullPath = path.join(charsDir, bwFilename);
         const bwExists = FileHandler.findInputFile(bwFullPath).found;

         if (bwExists) {
             console.error(`DEBUG - B&W Character Sheet already exists: ${bwFullPath}. Skipping generation.`);
             generatedFiles.push(bwFullPath);
             try {
                 bwRefBase64 = await FileHandler.readImageAsBase64(bwFullPath);
             } catch (e) {
                 console.error(`DEBUG - Failed to read existing B&W sheet for color ref:`, e);
             }
         } else {
             console.error(`DEBUG - Generating B&W Character Sheet for ${safeName}...`);
             const layoutPrompt = request.layout === 'strip' ? 'Wide Landscape 16:9' : (request.layout === 'webtoon' ? 'Tall Vertical 9:16' : (request.layout === 'single_page' ? 'Portrait 3:4' : 'Square 1:1'));
             const viewsPrompt = request.layout === 'strip' ? 
                `Include the following views: Front view, Left profile view, Right profile view, and Back view. Order them: Front, Left, Right, Back side-by-side.` : 
                `Composition: Split the image. Left Half: Full Body Standing Pose (Front View). Right Half Top: Close-up Face (Front View). Right Half Bottom: Back of Head/Upper Back View.`;

             const bwPrompt = `Character Design Sheet (${layoutPrompt}): ${sourceName}.
             Create a character design sheet with ${layoutPrompt} orientation.
             IMPORTANT: You MUST use the attached reference photo as the PRIMARY source for the character's physical appearance (face, body type, hair, clothing). The output character must look exactly like the person in the reference photo.
             ${characterDescription ? `Story Context: "${characterDescription}". Merge these traits with the visual reference, but prioritize the reference image for physical likeness.` : ''}
             ${viewsPrompt}
             Capture the facial features, hairstyle, and clothing details from the photo accurately but stylized.
             Determine the character's age category based on the reference image and apply the corresponding anatomical guidelines:
             - Child (approx 7-10): Head-to-body ratio 1:6, softer jawlines, shorter/slender limbs.
             - Adult (approx 25-40): Standard 1:7.5 to 1:8 head-to-body ratio, defined bone structure, balanced muscle tone.
             - Elder (70+): Slight natural spinal curvature (kyphosis), settled center of gravity, prominent joint articulation.
             ${request.style || 'shonen'} manga style, black and white, screentones, high quality line art.
             Full body from head to toe (must include complete legs and shoes), neutral pose, white background. DO NOT SQUASH or compress the figure vertically. Ensure legs are long and anatomically correct. Avoid chibi, dwarf, or super-deformed proportions. Zoom out to fit the entire character within the frame. Leave ample white space margin around the character to prevent cropping of feet or head.`;

             // Save B&W Prompt
             try {
                 await FileHandler.saveTextFile(path.join(promptsDir, `character_create_${safeName}_bw.txt`), bwPrompt);
             } catch (e) { console.error('DEBUG - Failed to save BW prompt:', e); }

             try {
                const bwResponse = await this.ai.models.generateContent({
                    model: this.modelName,
                    config: {
                      responseModalities: request.includeText ? ['IMAGE', 'TEXT'] : ['IMAGE'],
                      imageConfig: { aspectRatio: this.getAspectRatioString(request.layout) },
                      safetySettings: this.getSafetySettings(),
                    } as any,
                    contents: [{ role: 'user', parts: [
                        { text: bwPrompt },
                        { inlineData: { data: sourceB64, mimeType: 'image/png' } }
                    ] }],
                });

                if (bwResponse.candidates && bwResponse.candidates[0]?.content?.parts) {
                    for (const part of bwResponse.candidates[0].content.parts) {
                        let b64: string | undefined;
                        if (part.inlineData?.data) b64 = part.inlineData.data;
                        else if (part.text && this.isValidBase64ImageData(part.text)) b64 = part.text;

                        if (b64) {
                            const fullPath = await FileHandler.saveImageFromBase64(b64, charsDir, bwFilename);
                            generatedFiles.push(fullPath);
                            bwRefBase64 = b64;
                            await this.logGeneration(this.modelName, [fullPath], request.inputImage);
                            console.error(`DEBUG - Saved B&W Sheet: ${fullPath}`);
                            break;
                        }
                    }
                }
             } catch (e) {
                 console.error(`DEBUG - Failed to generate B&W sheet:`, e);
             }
         }

         // 2. Generate Color Sheet
         const colorFullPath = path.join(charsDir, colorFilename);
         const colorExists = FileHandler.findInputFile(colorFullPath).found;

         if (colorExists) {
             console.error(`DEBUG - Color Character Sheet already exists: ${colorFullPath}. Skipping generation.`);
             generatedFiles.push(colorFullPath);
         } else {
             console.error(`DEBUG - Generating Color Character Sheet for ${safeName}...`);
             const layoutPrompt = request.layout === 'strip' ? 'Wide Landscape 16:9' : (request.layout === 'webtoon' ? 'Tall Vertical 9:16' : (request.layout === 'single_page' ? 'Portrait 3:4' : 'Square 1:1'));
             const viewsPrompt = request.layout === 'strip' ? 
                `Include the following views: Front view, Left profile view, Right profile view, and Back view. Order them: Front, Left, Right, Back side-by-side.` : 
                `Composition: Split the image. Left Half: Full Body Standing Pose (Front View). Right Half Top: Close-up Face (Front View). Right Half Bottom: Back of Head/Upper Back View.`;

             const colorPrompt = `Character Design Sheet (${layoutPrompt}): ${sourceName}.
             Create a character design sheet with ${layoutPrompt} orientation.
             IMPORTANT: You MUST use the attached B&W character sheet as the STRICT reference for line art, pose, and design. Colorize it accurately.
             ${characterDescription ? `Story Context: "${characterDescription}".` : ''}
             The output must match the B&W reference exactly in terms of poses and layout.
             ${viewsPrompt}
             Capture the facial features, hairstyle, and clothing details from the reference accurately.
             GENERATE IN FULL COLOR. Vibrant colors, detailed shading.
             Anime/Manga style.
             Full body from head to toe (must include complete legs and shoes), neutral pose, white background. DO NOT SQUASH or compress the figure vertically. Ensure legs are long and anatomically correct. Avoid chibi, dwarf, or super-deformed proportions. Zoom out to fit the entire character within the frame. Leave ample white space margin around the character to prevent cropping of feet or head.`;

             // Save Color Prompt
             try {
                 await FileHandler.saveTextFile(path.join(promptsDir, `character_create_${safeName}_color.txt`), colorPrompt);
             } catch (e) { console.error('DEBUG - Failed to save Color prompt:', e); }

             try {
                const colorRefData = bwRefBase64 || sourceB64;
                const colorResponse = await this.ai.models.generateContent({
                    model: this.modelName,
                    config: {
                      responseModalities: request.includeText ? ['IMAGE', 'TEXT'] : ['IMAGE'],
                      imageConfig: { aspectRatio: this.getAspectRatioString(request.layout) },
                      safetySettings: this.getSafetySettings(),
                    } as any,
                    contents: [{ role: 'user', parts: [
                        { text: colorPrompt },
                        { inlineData: { data: colorRefData, mimeType: 'image/png' } }
                    ] }],
                });

                if (colorResponse.candidates && colorResponse.candidates[0]?.content?.parts) {
                    for (const part of colorResponse.candidates[0].content.parts) {
                        let b64: string | undefined;
                        if (part.inlineData?.data) b64 = part.inlineData.data;
                        else if (part.text && this.isValidBase64ImageData(part.text)) b64 = part.text;

                        if (b64) {
                            const fullPath = await FileHandler.saveImageFromBase64(b64, charsDir, colorFilename);
                            generatedFiles.push(fullPath);
                            await this.logGeneration(this.modelName, [fullPath], request.inputImage);
                            console.error(`DEBUG - Saved Color Sheet: ${fullPath}`);
                            break;
                        }
                    }
                }
             } catch (e) {
                 console.error(`DEBUG - Failed to generate Color sheet:`, e);
             }
         }

         if (generatedFiles.length > 0) {
             await this.handlePreview(generatedFiles, request);
             return {
                 success: true,
                 message: `Successfully created character sheets for ${sourceName} in ${charsDir}`,
                 generatedFiles
             };
         } else {
             return {
                 success: false,
                 message: 'Failed to generate character sheets.',
                 error: 'No images generated.'
             };
         }
      }

      // Handle Directory Batch Processing
      if (request.inputDirectory) {
        console.error(`DEBUG - Manga Directory Mode: Processing ${request.inputDirectory}`);
        const dirResult = FileHandler.findInputDirectory(request.inputDirectory);

        if (!dirResult.found || dirResult.files.length === 0) {
           return {
            success: false,
            message: `Input directory not found or empty: ${request.inputDirectory}`,
            error: 'Directory not found or empty',
          };
        }

        // Sort files to ensure sequence (using natural sort for numbered files)
        dirResult.files.sort((a, b) => 
          a.localeCompare(b, undefined, { numeric: true, sensitivity: 'base' })
        );

        console.error(`DEBUG - Found ${dirResult.files.length} images in directory.`);
        const generatedFiles: string[] = [];
        const errors: string[] = [];
        
        let characterReference: { data: string; mimeType: string } | null = null;
        let previousGeneratedImagePath: string | null = null;

        // Load Character Reference if provided
        if (request.characterImage) {
            const charRes = FileHandler.findInputFile(request.characterImage);
            if (charRes.found) {
                 try {
                     const b64 = await FileHandler.readImageAsBase64(charRes.filePath!);
                     characterReference = { data: b64, mimeType: 'image/png' };
                     console.error(`DEBUG - Loaded character reference: ${request.characterImage}`);
                 } catch (e) {
                     console.error(`DEBUG - Failed to load character reference:`, e);
                 }
            }
        }

        for (const filePath of dirResult.files) {
             console.error(`DEBUG - Processing file: ${filePath}`);
             try {
                const imageBase64 = await FileHandler.readImageAsBase64(filePath);
                
                let editPrompt = `${request.prompt}. Maintain the original composition and content structure while applying the requested style and details.`;
                
                if (request.color) {
                    editPrompt += " GENERATE IN FULL COLOR. Ignore any black and white instructions.";
                }

                if (characterReference) {
                    editPrompt += " Use the attached character reference image to strictly maintain consistent character colors (e.g., hair, clothes, eyes) and design.";
                }

                if (previousGeneratedImagePath) {
                    editPrompt += " Ensure the visual style and color grading matches the immediately preceding page (also attached) for continuity.";
                }

                const parts: any[] = [
                    { text: editPrompt },
                    { inlineData: { data: imageBase64, mimeType: 'image/png' } },
                ];

                // Attach references
                if (characterReference) {
                    parts.push({ inlineData: characterReference });
                }

                // Attach previous generated image for continuity
                if (previousGeneratedImagePath) {
                     try {
                        const prevB64 = await FileHandler.readImageAsBase64(previousGeneratedImagePath);
                        parts.push({ inlineData: { data: prevB64, mimeType: 'image/png' } });
                     } catch (e) {
                         console.error(`DEBUG - Failed to load previous generated image for ref:`, e);
                     }
                }
                
                const aspectRatio = this.getAspectRatioString(request.layout);
                console.error(`DEBUG - Generating with Aspect Ratio: ${aspectRatio}`);

                const response = await this.ai.models.generateContent({
                    model: this.modelName,
                    config: {
                      responseModalities: request.includeText ? ['IMAGE', 'TEXT'] : ['IMAGE'],
                      imageConfig: {
                        aspectRatio: this.getAspectRatioString(request.layout),
                      },
                      safetySettings: this.getSafetySettings(),
                    } as any,
                    contents: [
                      {
                        role: 'user',
                        parts: parts,
                      },
                    ],
                  });

                  if (response.candidates && response.candidates[0]?.content?.parts) {
                    for (const part of response.candidates[0].content.parts) {
                        let resultImageBase64: string | undefined;
                        if (part.inlineData?.data) {
                            resultImageBase64 = part.inlineData.data;
                        } else if (part.text && this.isValidBase64ImageData(part.text)) {
                            resultImageBase64 = part.text;
                        }

                        if (resultImageBase64) {
                            // Use original filename as base
                            const originalName = filePath.split(/[/\\]/).pop()?.split('.')[0] || 'image';
                            const filename = FileHandler.generateFilename(
                                `manga_edit_${originalName}`,
                                'png',
                                0,
                            );
                            const fullPath = await FileHandler.saveImageFromBase64(
                                resultImageBase64,
                                FileHandler.ensureOutputDirectory(),
                                filename,
                            );
                            generatedFiles.push(fullPath);
                            await this.logGeneration(this.modelName, [fullPath]);
                            previousGeneratedImagePath = fullPath; // Set for next iteration
                            console.error(`DEBUG - Saved: ${fullPath}`);
                            break; 
                        }
                    }
                  }
             } catch (e) {
                 const msg = `Failed to process ${filePath}: ${e instanceof Error ? e.message : String(e)}`;
                 console.error(msg);
                 errors.push(msg);
             }
        }

        if (generatedFiles.length > 0) {
            await this.handlePreview(generatedFiles, request);
            return {
                success: true,
                message: `Successfully processed ${generatedFiles.length} images from directory.`,
                generatedFiles,
            };
        } else {
            return {
                success: false,
                message: 'Failed to process any images in directory.',
                error: errors.join('\n'),
            };
        }
      }

      // Handle Image Editing Mode (if inputImage is provided)
      if (request.inputImage) {
        console.error(`DEBUG - Manga Edit Mode: Processing ${request.inputImage}`);
        
        const fileResult = FileHandler.findInputFile(request.inputImage);
        if (!fileResult.found) {
          return {
            success: false,
            message: `Input image not found: ${request.inputImage}`,
            error: `Searched in: ${fileResult.searchedPaths.join(', ')}`,
          };
        }

        const outputPath = FileHandler.ensureOutputDirectory();
        const imageBase64 = await FileHandler.readImageAsBase64(fileResult.filePath!);

        // Construct a specialized prompt for editing if needed, or use the one from request
        // The request.prompt is already built by buildMangaPrompt with style/layout info
        const editPrompt = `${request.prompt}. Maintain the original composition and content structure while applying the requested style and details.`;

        try {
          const aspectRatio = this.getAspectRatioString(request.layout);
          console.error(`DEBUG - Generating with Aspect Ratio: ${aspectRatio}`);

          const response = await this.ai.models.generateContent({
            model: this.modelName,
            config: {
              responseModalities: request.includeText ? ['IMAGE', 'TEXT'] : ['IMAGE'],
              imageConfig: {
                aspectRatio: this.getAspectRatioString(request.layout),
              },
              safetySettings: this.getSafetySettings(),
            } as any,
            contents: [
              {
                role: 'user',
                parts: [
                  { text: editPrompt },
                  {
                    inlineData: {
                      data: imageBase64,
                      mimeType: 'image/png',
                    },
                  },
                ],
              },
            ],
          });

          if (response.candidates && response.candidates[0]?.content?.parts) {
            for (const part of response.candidates[0].content.parts) {
              let resultImageBase64: string | undefined;

              if (part.inlineData?.data) {
                resultImageBase64 = part.inlineData.data;
              } else if (part.text && this.isValidBase64ImageData(part.text)) {
                resultImageBase64 = part.text;
              }

              if (resultImageBase64) {
                const filename = FileHandler.generateFilename(
                  `manga_edit_${request.prompt}`,
                  'png',
                  0,
                );
                const fullPath = await FileHandler.saveImageFromBase64(
                  resultImageBase64,
                  outputPath,
                  filename,
                );
                await this.logGeneration(this.modelName, [fullPath]);
                console.error('DEBUG - Edited manga page saved to:', fullPath);
                
                await this.handlePreview([fullPath], request);

                return {
                  success: true,
                  message: `Successfully edited manga page`,
                  generatedFiles: [fullPath],
                };
              }
            }
          }
          return {
              success: false,
              message: 'Failed to generate edited manga image',
              error: 'No image data in response',
          };

        } catch (error: unknown) {
           return {
            success: false,
            message: 'Failed to edit manga page',
            error: this.handleApiError(error),
          };
        }
      }

      // Existing Story Generation Logic
      if (!request.storyFile) {
        return {
          success: false,
          message: 'Story file is required for manga generation (unless --image is provided)',
          error: 'Missing storyFile parameter',
        };
      }

      const ratioInstruction = this.getAspectRatioInstruction(request.layout);
      console.error(`DEBUG - Adding aspect ratio instruction: ${ratioInstruction}`);

      // Read story file
      const storyFileResult = FileHandler.findInputFile(request.storyFile);
      if (!storyFileResult.found) {
        return {
          success: false,
          message: `Story file not found: ${request.storyFile}`,
          error: `Searched in: ${storyFileResult.searchedPaths.join(', ')}`,
        };
      }
      const storyContent = await FileHandler.readTextFile(
        storyFileResult.filePath!,
      );

      // Memory File Setup
      const memoryPath = await MemoryHandler.getMemoryFilePath(storyFileResult.filePath!);
      
      // Prompts Directory Setup (Global for this story)
      const promptsDir = path.join(path.dirname(storyFileResult.filePath!), 'prompts');
      FileHandler.ensureDirectory(promptsDir);

      // Parse Story Content for Pages
      // Splits by headers like "# Page 1", "## Page 2", "Page 3:"
      // CAPTURE THE FULL LINE to avoid title text leaking into content
      const pageRegex = /(?:^|\n)((?:#{1,3}\s*Page\s*\d+|Page\s*\d+:)[^\n]*)/i;
      const sections = storyContent.split(pageRegex);
      
      let globalContext = '';
      const pages: { header: string; content: string }[] = [];

      if (sections.length < 2) {
        // No distinct page headers found, treat as single page
        globalContext = ''; 
        pages.push({ header: 'Single Page', content: storyContent });
      } else {
        globalContext = sections[0].trim(); // Text before first page header
        for (let i = 1; i < sections.length; i += 2) {
          pages.push({
            header: sections[i].trim(),
            content: sections[i + 1] || '',
          });
        }
      }

      console.error(
        `DEBUG - Parsed ${pages.length} page(s) from story file. Global context length: ${globalContext.length}`,
      );

      const generatedFiles: string[] = [];
      let previousPagePath: string | null = null;
      let firstError: string | null = null;

      // Extract Global Reference Images
      const globalImagePaths = this.extractImagePaths(globalContext);
      const globalReferenceImages: { data: string; mimeType: string; sourcePath: string }[] = [];
      const loadedGlobalImagePaths = new Set<string>();

      for (const imgPath of globalImagePaths) {
        const fileRes = FileHandler.findInputFile(imgPath);
        if (fileRes.found) {
            try {
                // Avoid duplicates if multiple references point to same file
                if (loadedGlobalImagePaths.has(fileRes.filePath!)) continue;

                const b64 = await FileHandler.readImageAsBase64(fileRes.filePath!);
                globalReferenceImages.push({
                    data: b64,
                    mimeType: 'image/png', // Assuming png for simplicity, logic could be smarter
                    sourcePath: imgPath
                });
                loadedGlobalImagePaths.add(fileRes.filePath!);
                console.error(`DEBUG - Loaded global reference: ${imgPath}`);
            } catch (e) {
                console.error(`DEBUG - Failed to load global ref ${imgPath}:`, e);
            }
        }
      }

      // Explicit Character Image (CLI Argument)
      if (request.characterImage) {
        const charRes = FileHandler.findInputFile(request.characterImage);
        if (charRes.found && !loadedGlobalImagePaths.has(charRes.filePath!)) {
             const b64 = await FileHandler.readImageAsBase64(charRes.filePath!);
             globalReferenceImages.push({ data: b64, mimeType: 'image/png', sourcePath: request.characterImage });
             loadedGlobalImagePaths.add(charRes.filePath!);
             globalContext += `\n\n(See attached character reference: ${request.characterImage})`;
        }
      } else if (request.storyFile) {
        // Auto-detect or Generate Character Reference
        const storyFilename = request.storyFile.split(/[/\\]/).pop()?.split('.')[0] || 'story';
        const storyAbsPath = storyFileResult.filePath!;
        const storyDir = path.dirname(storyAbsPath);
        const charsDir = path.join(storyDir, 'characters');
        FileHandler.ensureDirectory(charsDir);
        
        const promptsDir = path.join(storyDir, 'prompts');
        FileHandler.ensureDirectory(promptsDir);

        // 1. Parse Explicit Character List in Text
        // Matches: * **Name:** Description  OR  - **Name:** Description
        // UPDATED: Now supports multi-line nested bullets (e.g. Visuals: ...)
        const charListRegex = /^\s*[\*\-]\s*\*\*([^\*:\n]+)(?::)?\*\*(.*)$/gm;
        let charMatch;
        let storyContentModified = false;
        let newStoryContent = storyContent;

        // We process matches from the original content to ensure we can replace lines accurately
        // Note: regex.exec is stateful.
        while ((charMatch = charListRegex.exec(globalContext)) !== null) {
            const fullMatchLine = charMatch[0];
            const charName = charMatch[1].trim();
            let charDesc = charMatch[2].trim();
            
            // Handle Multi-line/Nested Descriptions
            // If the description on the same line is empty or very short, look for nested bullets
            if (charDesc.length < 5) {
                const nextLinesStartIndex = charMatch.index + charMatch[0].length;
                const upcomingText = globalContext.substring(nextLinesStartIndex);
                
                // Look ahead for nested details
                 // 1000 chars should be enough window to find the description
                 const searchWindow = upcomingText.substring(0, 1000);
                 
                 // Aggregate all nested bullets until the next character definition
                 const lines = searchWindow.split('\n');
                 const descLines: string[] = [];
                 
                 // Get indentation of the parent character line to determine nesting
                 const parentIndent = fullMatchLine.match(/^\s*/)?.[0].length || 0;

                 for (const line of lines) {
                     if (line.trim() === '') continue; // skip empty lines?
                     
                     // Stop if we hit a new top-level character definition or section
                     if (line.match(/^\s*[\*\-]\s*\*\*/)) {
                         const lineIndent = line.match(/^\s*/)?.[0].length || 0;
                         if (lineIndent <= parentIndent) {
                             break;
                         }
                     }
                     if (line.match(/^#/)) break; // Section header
                     
                     // Clean up bullet points
                     // We try to remove common property keys to keep the description clean, but keeping them is also fine.
                     const cleaned = line.replace(/^\s*[\*\-]\s*(?:\*\*(?:Role|Vibe|Personality|Visuals|Appearance|Traits|Outfit|Features):\*\*)?/g, '').trim();
                     if (cleaned) descLines.push(cleaned);
                     
                     // Safety break to prevent reading too much
                     if (descLines.length > 20) break;
                 }
                 if (descLines.length > 0) charDesc = descLines.join(' ');
            }
            
            // Section Validation: Check if we are inside a "Character" section
            const matchIndex = charMatch.index;
            const textBeforeMatch = globalContext.substring(0, matchIndex);
            // Find the last header (# Header) before this match
            const lastHeaderMatch = [...textBeforeMatch.matchAll(/^(#{1,6})\s+(.*)$/gm)].pop();
            
            if (lastHeaderMatch) {
                const headerTitle = lastHeaderMatch[2].toLowerCase();
                // Valid sections must explicitly relate to characters
                const validSectionKeywords = ['character', 'cast', 'person', 'role', 'protagonist', 'antagonist'];
                let isValidSection = validSectionKeywords.some(kw => headerTitle.includes(kw));
                
                // FALLBACK: If the formal header didn't match (e.g. it was just the Story Title),
                // check for "Label-style" headers in the text between the header and this match.
                // Example: "# Title" ...text... "**Characters:**" ... * **Elara:**
                if (!isValidSection) {
                    const headerIndex = lastHeaderMatch.index! + lastHeaderMatch[0].length;
                    const textBetweenHeaderAndMatch = globalContext.substring(headerIndex, matchIndex);
                    
                    // Look for lines that look like "Characters:", "**Characters:**", "Cast:"
                    // We check for the keywords again but in a "label" context
                    const labelRegex = /(?:^|\n)\s*(?:\*\*|__)?\s*(?:Characters?|Cast|Roles?|Personas?)\s*(?:\*\*|__)?\s*[:\n]/i;
                    if (labelRegex.test(textBetweenHeaderAndMatch)) {
                        isValidSection = true;
                         console.error(`DEBUG - Validated character "${charName}" via text label (e.g. "**Characters:**")`);
                    }
                }
                
                if (!isValidSection) {
                    console.error(`DEBUG - Skipping potential character "${charName}" because it is in non-character section: "${lastHeaderMatch[2]}"`);
                    continue;
                }
            } else {
                // If no header found yet, maybe check if the name itself looks like a rule?
                // But generally safe to skip if we want strict sectioning.
                // For now, let's process if no header, but maybe filter ALL CAPS names longer than 2 words?
                if (charName === charName.toUpperCase() && charName.split(' ').length > 2) {
                     console.error(`DEBUG - Skipping "${charName}" (looks like a rule/instruction)`);
                     continue;
                }
            }

            const safeName = charName.toLowerCase().replace(/[^a-z0-9]/g, '_');
            const charFilename = request.color ? `${safeName}_portrait_color.png` : `${safeName}_portrait.png`;
            const charRelPath = path.join('characters', charFilename);
            const charAbsPath = path.join(charsDir, charFilename);

            // Skip if line already has THIS specific image link AND the file actually exists
            if (fullMatchLine.includes(charFilename)) {
                // IMPORTANT: Check if file exists on disk
                const existingRes = FileHandler.findInputFile(charAbsPath);
                if (existingRes.found) {
                    // Check for duplicates
                    if (loadedGlobalImagePaths.has(existingRes.filePath!)) {
                        console.error(`DEBUG - Character reference for ${charName} already loaded via global scan.`);
                        continue; 
                    }

                    try {
                        const b64 = await FileHandler.readImageAsBase64(existingRes.filePath!);
                        globalReferenceImages.push({ data: b64, mimeType: 'image/png', sourcePath: existingRes.filePath! });
                        loadedGlobalImagePaths.add(existingRes.filePath!);
                        console.error(`DEBUG - Loaded existing reference from text: ${charName}`);
                    } catch (e) {
                        console.error(`DEBUG - Failed to load existing reference for ${charName}:`, e);
                    }
                    continue; 
                } else {
                     console.error(`DEBUG - Character link found for ${charName} but file missing. Regenerating...`);
                }
            }

            console.error(`DEBUG - Processing character definition: ${charName} for ${request.color ? 'Color' : 'B&W'}`);
            
            let charFullPath = '';
            
            // Check for explicit image reference in the description FIRST
            const explicitRefs = this.extractImagePaths(charDesc);
            let foundExplicitRef = false;
            
            for (const refPath of explicitRefs) {
                 let refRes = FileHandler.findInputFile(refPath);
                 // Try relative to story directory if not found
                 if (!refRes.found) {
                     const relPath = path.join(storyDir, refPath);
                     const relRes = FileHandler.findInputFile(relPath);
                     if (relRes.found) refRes = relRes;
                 }

                 if (refRes.found) {
                     charFullPath = refRes.filePath!;
                     console.error(`DEBUG - Found explicit reference for ${charName} in description: ${charFullPath}`);
                     foundExplicitRef = true;
                     break; 
                 }
            }
            
            if (foundExplicitRef) {
                // We found one, so we skip standard check/generation
            } else {
                // Check if exists
                const existingRes = FileHandler.findInputFile(charAbsPath);
                if (existingRes.found) {
                    charFullPath = existingRes.filePath!;
                    console.error(`DEBUG - Found existing ref for ${charName}: ${charFullPath}`);
                } else {
                    // STRATEGY: Ensure BOTH B&W and Color versions exist to guarantee consistency for future runs.
                    // 1. Ensure B&W exists (Generate if missing)
                    // 2. Ensure Color exists (Generate from B&W if missing)
                    // 3. Return the one requested by current mode
                    
                    const bwFilename = `${safeName}_portrait.png`;
                    const bwAbsPath = path.join(charsDir, bwFilename);
                    const colorFilename = `${safeName}_portrait_color.png`;
                    const colorAbsPath = path.join(charsDir, colorFilename);
                    
                    let sourceBwImageBase64: string | undefined = undefined;

                    // Step 1: Handle B&W
                    const bwRes = FileHandler.findInputFile(bwAbsPath);
                    if (bwRes.found) {
                         try {
                             sourceBwImageBase64 = await FileHandler.readImageAsBase64(bwRes.filePath!);
                         } catch (e) { console.error(`DEBUG - Failed to read existing B&W ref:`, e); }
                    } else if (request.autoGenerateCharacters) {
                         // NEW: Check for Source Image in Description
                         const sourceImagePaths = this.extractImagePaths(charDesc);
                         let sourceImageB64: string | undefined = undefined;
                         
                         if (sourceImagePaths.length > 0) {
                            const sourceRes = FileHandler.findInputFile(sourceImagePaths[0]);
                            if (sourceRes.found) {
                                try {
                                    sourceImageB64 = await FileHandler.readImageAsBase64(sourceRes.filePath!);
                                    console.error(`DEBUG - Found source image for ${charName}: ${sourceRes.filePath}`);
                                } catch(e) { console.error(`DEBUG - Failed to read source image:`, e); }
                            }
                         }

                         console.error(`DEBUG - Generating BASE B&W ref for ${charName}...`);
                         const layoutPrompt = request.layout === 'strip' ? 'Wide Landscape 16:9' : (request.layout === 'webtoon' ? 'Tall Vertical 9:16' : (request.layout === 'single_page' ? 'Portrait 3:4' : 'Square 1:1'));
                         const viewsPrompt = request.layout === 'strip' ? 
                            `Include the following views: Front view, Left profile view, Right profile view, and Back view. Order them: Front, Left, Right, Back side-by-side.` : 
                            `Composition: Split the image. Left Half: Full Body Standing Pose (Front View). Right Half Top: Close-up Face (Front View). Right Half Bottom: Back of Head/Upper Back View.`;

                         const bwPrompt = `Character Design Sheet (${layoutPrompt}): ${charName}. ${charDesc}. 
                         ${viewsPrompt}
                         Ensure the character appeal and details strictly follow the guidelines provided in the user story file description.
                         Determine the character's age category based on the description and apply the corresponding anatomical guidelines:
                         - Child (approx 7-10): Head-to-body ratio 1:6, softer jawlines, shorter/slender limbs.
                         - Adult (approx 25-40): Standard 1:7.5 to 1:8 head-to-body ratio, defined bone structure, balanced muscle tone.
                         - Elder (70+): Slight natural spinal curvature (kyphosis), settled center of gravity, prominent joint articulation.
                         ${sourceImageB64 ? 'Use the attached image as the visual source for the character\'s appearance.' : ''}
                         ${request.style || 'shonen'} manga style, black and white, screentones, high quality line art.
                                 Full body from head to toe (must include complete legs and shoes), neutral pose, white background. DO NOT SQUASH or compress the figure vertically. Ensure legs are long and anatomically correct. Avoid chibi, dwarf, or super-deformed proportions. Zoom out to fit the entire character within the frame. Leave ample white space margin around the character to prevent cropping of feet or head.`;
                                         
                                         // Save B&W Prompt
                                         try {
                                             await FileHandler.saveTextFile(path.join(promptsDir, `character_create_${safeName}_bw.txt`), bwPrompt);
                                         } catch (e) { console.error('DEBUG - Failed to save BW prompt:', e); }
                         
                                         const bwParts: any[] = [{ text: bwPrompt }];                         if (sourceImageB64) {
                            bwParts.push({ inlineData: { data: sourceImageB64, mimeType: 'image/png' } });
                         }

                         try {
                            const bwResponse = await this.ai.models.generateContent({
                                model: this.modelName,
                                config: {
                                  responseModalities: request.includeText ? ['IMAGE', 'TEXT'] : ['IMAGE'],
                                  imageConfig: { aspectRatio: this.getAspectRatioString(request.layout) },
                                  safetySettings: this.getSafetySettings(),
                                } as any,
                                contents: [{ role: 'user', parts: bwParts }],
                            });

                            if (bwResponse.candidates && bwResponse.candidates[0]?.content?.parts) {
                                for (const part of bwResponse.candidates[0].content.parts) {
                                    let b64: string | undefined;
                                    if (part.inlineData?.data) b64 = part.inlineData.data;
                                    else if (part.text && this.isValidBase64ImageData(part.text)) b64 = part.text;

                                    if (b64) {
                                        const fullPath = await FileHandler.saveImageFromBase64(b64, charsDir, bwFilename);
                                        generatedFiles.push(fullPath);
                                        sourceBwImageBase64 = b64;
                                        console.error(`DEBUG - Generated Base B&W ${charName}: ${bwAbsPath}`);
                                        break;
                                    }
                                }
                            }
                         } catch (e) {
                             console.error(`DEBUG - Failed to generate base B&W character ${charName}:`, e);
                         }
                    } else {
                         console.error(`DEBUG - Skipping B&W generation for ${charName} (autoGenerateCharacters=false)`);
                    }

                    // Step 2: Handle Color (only if we have a B&W base to work from)
                    if (sourceBwImageBase64) {
                        const colorRes = FileHandler.findInputFile(colorAbsPath);
                        if (!colorRes.found) {
                            if (request.autoGenerateCharacters) {
                                console.error(`DEBUG - Generating Color ref for ${charName} (using B&W base)...`);
                                const layoutPrompt = request.layout === 'strip' ? 'Wide Landscape 16:9' : (request.layout === 'webtoon' ? 'Tall Vertical 9:16' : (request.layout === 'single_page' ? 'Portrait 3:4' : 'Square 1:1'));
                                const viewsPrompt = request.layout === 'strip' ? 
                                    `Include the following views: Front view, Left profile view, Right profile view, and Back view. Order them: Front, Left, Right, Back side-by-side.` : 
                                    `Composition: Split the image. Left Half: Full Body Standing Pose (Front View). Right Half Top: Close-up Face (Front View). Right Half Bottom: Back of Head/Upper Back View.`;

                                const colorPrompt = `Character Design Sheet (${layoutPrompt}): ${charName}. ${charDesc}. 
                                GENERATE IN FULL COLOR. Vibrant colors, detailed shading.
                                Use the attached B&W image as the STRICT reference for line art and design. Colorize it accurately.
                                ${viewsPrompt}
                                Ensure the character appeal and details strictly follow the guidelines provided in the user story file description.
                                Full body from head to toe (must include complete legs and shoes), neutral pose, white background. DO NOT SQUASH or compress the figure vertically. Ensure legs are long and anatomically correct. Avoid chibi, dwarf, or super-deformed proportions. Zoom out to fit the entire character within the frame. Leave ample white space margin around the character to prevent cropping of feet or head.`;

                                try {
                                    const colorResponse = await this.ai.models.generateContent({
                                        model: this.modelName,
                                        config: {
                                          responseModalities: request.includeText ? ['IMAGE', 'TEXT'] : ['IMAGE'],
                                          imageConfig: { aspectRatio: this.getAspectRatioString(request.layout) },
                                          safetySettings: this.getSafetySettings(),
                                        } as any,
                                        contents: [{ role: 'user', parts: [
                                            { text: colorPrompt },
                                            { inlineData: { data: sourceBwImageBase64, mimeType: 'image/png' } }
                                        ]}],
                                    });

                                    if (colorResponse.candidates && colorResponse.candidates[0]?.content?.parts) {
                                        for (const part of colorResponse.candidates[0].content.parts) {
                                            let b64: string | undefined;
                                            if (part.inlineData?.data) b64 = part.inlineData.data;
                                            else if (part.text && this.isValidBase64ImageData(part.text)) b64 = part.text;

                                            if (b64) {
                                                const fullPath = await FileHandler.saveImageFromBase64(b64, charsDir, colorFilename);
                                                generatedFiles.push(fullPath);
                                                console.error(`DEBUG - Generated Color ${charName}: ${colorAbsPath}`);
                                                break;
                                            }
                                        }
                                    }
                                } catch (e) {
                                     console.error(`DEBUG - Failed to generate color character ${charName}:`, e);
                                }
                            } else {
                                console.error(`DEBUG - Skipping Color generation for ${charName} (autoGenerateCharacters=false)`);
                            }
                        }
                    }

                    // Step 3: Set charFullPath to the requested one
                    if (request.color) {
                        // We just tried to ensure it exists
                         const check = FileHandler.findInputFile(colorAbsPath);
                         if (check.found) charFullPath = check.filePath!;
                    } else {
                         const check = FileHandler.findInputFile(bwAbsPath);
                         if (check.found) charFullPath = check.filePath!;
                    }
                }
            }

            if (charFullPath) {
                // Add to current session refs
                if (loadedGlobalImagePaths.has(charFullPath)) {
                    // Already loaded
                } else {
                    try {
                        const b64 = await FileHandler.readImageAsBase64(charFullPath);
                        globalReferenceImages.push({ data: b64, mimeType: 'image/png', sourcePath: charFullPath });
                        loadedGlobalImagePaths.add(charFullPath);
                        // Append link to the story file content
                        // We need to determine the correct relative path based on what was selected
                        const finalFilename = request.color ? `${safeName}_portrait_color.png` : `${safeName}_portrait.png`;
                        const finalRelPath = path.join('characters', finalFilename);
                        
                        newStoryContent = newStoryContent.replace(fullMatchLine, `${fullMatchLine} ![${charName}](${finalRelPath})`);
                        storyContentModified = true;
                    } catch (e) {
                        console.error(`DEBUG - Error processing ${charName} image:`, e);
                    }
                }
            }
        }

        // 1.5 Parse Header-based Character Definitions (### Name)
        // Matches: ### Name
        // Content...
        const headerCharRegex = /(?:^|\n)#{3}\s*(?!Page)([^:\n]+)(?::)?(?:\n|$)([\s\S]*?)(?=(?:\n#{1,3}|\n#{1,2}\s*Page|$))/gi;
        let headerMatch;
        
        while ((headerMatch = headerCharRegex.exec(globalContext)) !== null) {
            const charName = headerMatch[1].trim();
            const charDesc = headerMatch[2].trim();
            
            // Skip empty or likely system headers
            if (!charName || charName.toLowerCase() === 'character style' || charName.toLowerCase() === 'characters') continue;

            // Section Validation: Check if we are inside a "Character" section
            const matchIndex = headerMatch.index;
            const textBeforeMatch = globalContext.substring(0, matchIndex);
            // Find the last header that is NOT a level 3 header (so # or ##)
            // This is a simplification; we assume the parent section defines the context.
            const lastParentHeaderMatch = [...textBeforeMatch.matchAll(/^(#{1,2})\s+(.*)$/gm)].pop();

            if (lastParentHeaderMatch) {
                const headerTitle = lastParentHeaderMatch[2].toLowerCase();
                const validSectionKeywords = ['character', 'cast', 'person', 'role', 'protagonist', 'antagonist'];
                const isValidSection = validSectionKeywords.some(kw => headerTitle.includes(kw));
                
                if (!isValidSection) {
                     console.error(`DEBUG - Skipping potential character header "${charName}" because parent section is: "${lastParentHeaderMatch[2]}"`);
                     continue;
                }
            } else {
                 // If no parent header (top level ###?), risky. 
                 // But typically characters are under a ## Characters section.
                 // We'll skip if it looks like a Rule or Instruction header
                 const invalidKeywords = ['rule', 'prompt', 'instruction', 'setting', 'format', 'optimization', 'export'];
                 if (invalidKeywords.some(kw => charName.toLowerCase().includes(kw))) {
                     console.error(`DEBUG - Skipping header "${charName}" (looks like instruction)`);
                     continue;
                 }
            }

            const safeName = charName.toLowerCase().replace(/[^a-z0-9]/g, '_');
            const charFilename = request.color ? `${safeName}_portrait_color.png` : `${safeName}_portrait.png`;
            const charRelPath = path.join('characters', charFilename);
            const charAbsPath = path.join(charsDir, charFilename);
            
            // Skip if description already contains the target link (to avoid loops or double adds)
            if (charDesc.includes(charFilename)) {
                // IMPORTANT: Even if the link exists, we MUST load it into memory for this session
                const existingRes = FileHandler.findInputFile(charAbsPath);
                if (existingRes.found) {
                    if (loadedGlobalImagePaths.has(existingRes.filePath!)) {
                        continue;
                    }
                    try {
                        const b64 = await FileHandler.readImageAsBase64(existingRes.filePath!);
                        globalReferenceImages.push({ data: b64, mimeType: 'image/png', sourcePath: existingRes.filePath! });
                        loadedGlobalImagePaths.add(existingRes.filePath!);
                        console.error(`DEBUG - Loaded existing reference from text (Header): ${charName}`);
                    } catch (e) {
                        console.error(`DEBUG - Failed to load existing reference for ${charName}:`, e);
                    }
                }
                continue;
            }

            console.error(`DEBUG - Processing character definition (Header): ${charName}`);
            
            let charFullPath = '';
            
            // Check for explicit image reference in the description FIRST
            const explicitRefs = this.extractImagePaths(charDesc);
            let foundExplicitRef = false;
            
            for (const refPath of explicitRefs) {
                 let refRes = FileHandler.findInputFile(refPath);
                 // Try relative to story directory if not found
                 if (!refRes.found) {
                     const relPath = path.join(storyDir, refPath);
                     const relRes = FileHandler.findInputFile(relPath);
                     if (relRes.found) refRes = relRes;
                 }

                 if (refRes.found) {
                     charFullPath = refRes.filePath!;
                     console.error(`DEBUG - Found explicit reference for ${charName} in description: ${charFullPath}`);
                     foundExplicitRef = true;
                     break; 
                 }
            }
            
            if (foundExplicitRef) {
                // Skip generation
            } else {
                // Check if exists standard named file
                const existingRes = FileHandler.findInputFile(charAbsPath);
                if (existingRes.found) {
                    charFullPath = existingRes.filePath!;
                    console.error(`DEBUG - Found existing ref for ${charName}: ${charFullPath}`);
                } else {
                     // Generation Logic (Same as above)
                    const bwFilename = `${safeName}_portrait.png`;
                    const bwAbsPath = path.join(charsDir, bwFilename);
                    const colorFilename = `${safeName}_portrait_color.png`;
                    const colorAbsPath = path.join(charsDir, colorFilename);
                    
                    let sourceBwImageBase64: string | undefined = undefined;

                    // Step 1: Handle B&W
                    const bwRes = FileHandler.findInputFile(bwAbsPath);
                    if (bwRes.found) {
                         try { sourceBwImageBase64 = await FileHandler.readImageAsBase64(bwRes.filePath!); } catch (e) {}
                    } else if (request.autoGenerateCharacters) {
                         console.error(`DEBUG - Generating BASE B&W ref for ${charName}...`);
                                                  const layoutPrompt = request.layout === 'strip' ? 'Wide Landscape 16:9' : (request.layout === 'webtoon' ? 'Tall Vertical 9:16' : (request.layout === 'single_page' ? 'Portrait 3:4' : 'Square 1:1'));
                                                  const viewsPrompt = request.layout === 'strip' ? 
                                                    `Include the following views: Front view, Left profile view, Right profile view, and Back view. Order them: Front, Left, Right, Back side-by-side.` : 
                                                    `Composition: Split the image. Left Half: Full Body Standing Pose (Front View). Right Half Top: Close-up Face (Front View). Right Half Bottom: Back of Head/Upper Back View.`;

                                                  const bwPrompt = `Character Design Sheet (${layoutPrompt}): ${charName}. ${charDesc}.
                                                  ${viewsPrompt}
                                                  Ensure the character appeal and details strictly follow the guidelines provided in the user story file description.
                                                  Determine the character's age category based on the description and apply the corresponding anatomical guidelines:
                                                  - Child (approx 7-10): Head-to-body ratio 1:6, softer jawlines, shorter/slender limbs.
                                                  - Adult (approx 25-40): Standard 1:7.5 to 1:8 head-to-body ratio, defined bone structure, balanced muscle tone.
                                                  - Elder (70+): Slight natural spinal curvature (kyphosis), settled center of gravity, prominent joint articulation.
                                                  ${request.style || 'shonen'} manga style, black and white, screentones, high quality line art.
                                                  Full body from head to toe (must include complete legs and shoes), neutral pose, white background. DO NOT SQUASH or compress the figure vertically. Ensure legs are long and anatomically correct. Avoid chibi, dwarf, or super-deformed proportions. Zoom out to fit the entire character within the frame. Leave ample white space margin around the character to prevent cropping of feet or head.`;                         
                         try {
                            const bwResponse = await this.ai.models.generateContent({
                                model: this.modelName,
                                config: {
                                  responseModalities: request.includeText ? ['IMAGE', 'TEXT'] : ['IMAGE'],
                                  imageConfig: { aspectRatio: this.getAspectRatioString(request.layout) },
                                  safetySettings: this.getSafetySettings(),
                                } as any,
                                contents: [{ role: 'user', parts: [{ text: bwPrompt }] }],
                            });
                            // Save logic...
                            if (bwResponse.candidates && bwResponse.candidates[0]?.content?.parts) {
                                for (const part of bwResponse.candidates[0].content.parts) {
                                    let b64: string | undefined;
                                    if (part.inlineData?.data) b64 = part.inlineData.data;
                                    else if (part.text && this.isValidBase64ImageData(part.text)) b64 = part.text;
                                    if (b64) {
                                        const fullPath = await FileHandler.saveImageFromBase64(b64, charsDir, bwFilename);
                                        generatedFiles.push(fullPath);
                                        sourceBwImageBase64 = b64;
                                        console.error(`DEBUG - Generated Base B&W ${charName}: ${bwAbsPath}`);
                                        break;
                                    }
                                }
                            }
                         } catch (e) { console.error(`DEBUG - Failed to generate base B&W character ${charName}:`, e); }
                    } else {
                         console.error(`DEBUG - Skipping B&W generation for ${charName} (autoGenerateCharacters=false)`);
                    }

                    // Step 2: Handle Color (only if B&W exists)
                    if (sourceBwImageBase64) {
                        const colorRes = FileHandler.findInputFile(colorAbsPath);
                        if (!colorRes.found) {
                            if (request.autoGenerateCharacters) {
                                console.error(`DEBUG - Generating Color ref for ${charName}...`);
                                const layoutPrompt = request.layout === 'strip' ? 'Wide Landscape 16:9' : (request.layout === 'webtoon' ? 'Tall Vertical 9:16' : (request.layout === 'single_page' ? 'Portrait 3:4' : 'Square 1:1'));
                                const viewsPrompt = request.layout === 'strip' ? 
                                    `Include the following views: Front view, Left profile view, Right profile view, and Back view. Order them: Front, Left, Right, Back side-by-side.` : 
                                    `Composition: Split the image. Left Half: Full Body Standing Pose (Front View). Right Half Top: Close-up Face (Front View). Right Half Bottom: Back of Head/Upper Back View.`;

                                const colorPrompt = `Character Design Sheet (${layoutPrompt}): ${charName}. ${charDesc}. 
                                GENERATE IN FULL COLOR. Vibrant colors, detailed shading.
                                Use the attached B&W image as the STRICT reference.
                                ${viewsPrompt}
                                Full body from head to toe (must include complete legs and shoes), neutral pose, white background. DO NOT SQUASH or compress the figure vertically. Ensure legs are long and anatomically correct. Avoid chibi, dwarf, or super-deformed proportions. Zoom out to fit the entire character within the frame. Leave ample white space margin around the character to prevent cropping of feet or head.`;

                                try {
                                    const colorResponse = await this.ai.models.generateContent({
                                        model: this.modelName,
                                        config: {
                                          responseModalities: request.includeText ? ['IMAGE', 'TEXT'] : ['IMAGE'],
                                          imageConfig: { aspectRatio: this.getAspectRatioString(request.layout) },
                                          safetySettings: this.getSafetySettings(),
                                        } as any,
                                        contents: [{ role: 'user', parts: [
                                            { text: colorPrompt },
                                            { inlineData: { data: sourceBwImageBase64, mimeType: 'image/png' } }
                                        ]}],
                                    });
                                    // Save logic...
                                    if (colorResponse.candidates && colorResponse.candidates[0]?.content?.parts) {
                                        for (const part of colorResponse.candidates[0].content.parts) {
                                            let b64: string | undefined;
                                            if (part.inlineData?.data) b64 = part.inlineData.data;
                                            else if (part.text && this.isValidBase64ImageData(part.text)) b64 = part.text;
                                            if (b64) {
                                                const fullPath = await FileHandler.saveImageFromBase64(b64, charsDir, colorFilename);
                                                generatedFiles.push(fullPath);
                                                console.error(`DEBUG - Generated Color ${charName}: ${colorAbsPath}`);
                                                break;
                                            }
                                        }
                                    }
                                } catch (e) {}
                            } else {
                                console.error(`DEBUG - Skipping Color generation for ${charName} (autoGenerateCharacters=false)`);
                            }
                        }
                    }

                    // Set Final Path
                    if (request.color) {
                         const check = FileHandler.findInputFile(colorAbsPath);
                         if (check.found) charFullPath = check.filePath!;
                    } else {
                         const check = FileHandler.findInputFile(bwAbsPath);
                         if (check.found) charFullPath = check.filePath!;
                    }
                }
            }

            if (charFullPath) {
                if (!loadedGlobalImagePaths.has(charFullPath)) {
                    try {
                        const b64 = await FileHandler.readImageAsBase64(charFullPath);
                        globalReferenceImages.push({ data: b64, mimeType: 'image/png', sourcePath: charFullPath });
                        loadedGlobalImagePaths.add(charFullPath);
                        console.error(`DEBUG - Registered ${charName} from header section.`);
                        
                        // Note: We don't auto-update the text file for Header sections as easily 
                        // because appending the link might break the visual structure or bullets.
                        // But since we registered it in globalReferenceImages, the generation will work.
                    } catch (e) {
                        console.error(`DEBUG - Error processing ${charName} image:`, e);
                    }
                }
            }
        }

        if (storyContentModified) {
            await FileHandler.saveTextFile(storyAbsPath, newStoryContent);
            console.error(`DEBUG - Updated story file with new character links.`);
            
            // Update globalContext variable so the rest of the prompt sees the links (though we already added the images manually)
            // Actually, we should probably update globalContext to match newStoryContent's relevant part
            // But since we pushed images to `globalReferenceImages`, the AI has the visual data.
            // The text context update is secondary but good for consistency.
            // Re-extracting globalContext from newStoryContent is safest but we can just append a note.
            globalContext += "\n(Note: Character references have been auto-generated and attached)";
        }

        // Check for Character Generation Only Mode
        if (request.characterGenerationOnly) {
             const successMsg = `Successfully generated/verified character sheets for story. Skipping page generation (Character Generation Only mode).`;
             console.error(`DEBUG - ${successMsg}`);
             return {
                 success: true,
                 message: successMsg,
                 generatedFiles
             };
        }

        // Handle Environment Generation
        if (request.autoGenerateEnvironments || request.environmentGenerationOnly) {
            console.error(`DEBUG - Auto-generating environments...`);
            const envsDir = path.join(storyDir, 'environments');
            FileHandler.ensureDirectory(envsDir);
            
            // Regex to find Environment/Setting sections
            // Matches: ## Environment Setup (Scene 1) ... content ...
            const envSectionRegex = /(?:^|\n)#{2,3}\s*(?:Environment|Setting|Location)s?\s*(?:Setup)?(?:\s*\(.*?\))?(?::)?(?:\n|$)([\s\S]*?)(?=(?:\n#{1,3}|\n#{1,2}\s*Page|$))/gi;
            
            // NEW: Regex to find "ENVIRONMENT ANCHORS" style lists (common in Global Prompt Rules)
            // Matches: - **ENVIRONMENT ANCHORS**: \n ... indented items ...
            const envAnchorsRegex = /(?:^|\n)\s*[\-\*]\s*\*\*(?:ENVIRONMENT|SETTING|LOCATION)(?:\s+ANCHORS?)?\*\*:(?:\s*\n)([\s\S]*?)(?=(?:\n\s*[\-\*]\s*\*\*|\n#|$))/gi;
            
            const allEnvSections: string[] = [];
            let match;
            let envStoryContentModified = false;
            
            // 1. Collect Header-based sections
            while ((match = envSectionRegex.exec(globalContext)) !== null) {
                console.error(`DEBUG - Found Environment Section (Header): ${match[0].trim().split('\n')[0]}`);
                allEnvSections.push(match[1]);
            }

            // 2. Collect List-based Anchors
            while ((match = envAnchorsRegex.exec(globalContext)) !== null) {
                console.error(`DEBUG - Found Environment Anchors (List): ENVIRONMENT ANCHORS`);
                allEnvSections.push(match[1]);
            }
            
            // NEW: Auto-Fix / Auto-Extract Environments if missing
            if (allEnvSections.length === 0) {
                 console.error(`DEBUG - No explicit Environment Setup found. Analyzing story to extract settings...`);
                 
                 const analysisPrompt = `Analyze the following manga story script. Identify the key recurring locations/settings (Environment Anchors) where the scenes take place.
                 
                 Generate a Markdown section titled "## Environment Setup".
                 Under it, list the environments as bullet points in this EXACT format:
                 - **Name**: Visual description of the setting, furniture, lighting, and atmosphere.
                 
                 Only include major locations that appear in the script. Keep descriptions visual and concise for an artist.
                 Do not include characters in the descriptions.
                 
                 STORY SCRIPT:
                 ${storyContent.substring(0, 15000)} // Limit context if needed
                 `;

                 try {
                     const extractionResponse = await this.ai.models.generateContent({
                        model: 'gemini-2.0-flash', // Use a text model for analysis if available, or fall back to image model which handles text too
                        config: {
                            responseModalities: ['TEXT'],
                            safetySettings: this.getSafetySettings(),
                        } as any,
                        contents: [{ role: 'user', parts: [{ text: analysisPrompt }] }],
                     });

                     if (extractionResponse.candidates && extractionResponse.candidates[0]?.content?.parts?.[0]?.text) {
                         const extractedText = extractionResponse.candidates[0].content.parts[0].text.trim();
                         console.error(`DEBUG - Extracted Environments:\n${extractedText}`);
                         
                         // Append to Story File
                         if (extractedText.includes('## Environment Setup') || extractedText.includes('**')) {
                             // Ensure we have a clean separation
                             const appendText = `\n\n${extractedText}`;
                             newStoryContent += appendText;
                             // We also need to update globalContext so the regex below finds it
                             globalContext += appendText;
                             envStoryContentModified = true;
                             
                             // Re-run regex to populate allEnvSections
                             let newMatch;
                             while ((newMatch = envSectionRegex.exec(globalContext)) !== null) {
                                // Avoid adding duplicates if regex finds old ones (unlikely given length check)
                                if (!allEnvSections.includes(newMatch[1])) {
                                    allEnvSections.push(newMatch[1]);
                                }
                             }
                             // Just in case the LLM used the List format (unlikely given prompt)
                             while ((newMatch = envAnchorsRegex.exec(globalContext)) !== null) {
                                 if (!allEnvSections.includes(newMatch[1])) {
                                     allEnvSections.push(newMatch[1]);
                                 }
                             }
                             
                             // If still empty, maybe the LLM output *is* the list itself without the header capture group catching it perfectly?
                             // Fallback: Use the raw extracted text as the section content if regex failed but text looks valid.
                             if (allEnvSections.length === 0 && extractedText.includes('**')) {
                                 allEnvSections.push(extractedText);
                             }
                         }
                     }
                 } catch (e) {
                     console.error(`DEBUG - Failed to auto-extract environments:`, e);
                 }
            }

            // Process all found sections
            for (const sectionContent of allEnvSections) {
                 // Parse bullet points: - **Name:** Description
                 const envItemRegex = /^\s*[\*\-]\s*\*\*([^\*:\n]+)(?::)?\*\*(.*)$/gm;
                 let envItemMatch;
                 
                 while ((envItemMatch = envItemRegex.exec(sectionContent)) !== null) {
                     const fullLine = envItemMatch[0];
                     const envName = envItemMatch[1].trim();
                     const envDesc = envItemMatch[2].trim();
                     
                     const safeName = envName.toLowerCase().replace(/[^a-z0-9]/g, '_');
                     const envFilename = `${safeName}_environment.png`;
                     const envRelPath = path.join('environments', envFilename);
                     const envAbsPath = path.join(envsDir, envFilename);
                     
                     // Check if file exists on disk (Check for ANY of the new multi-angle files or the legacy one)
                     // We will prioritize the new format: _env_far.png
                     
                     // ORDER CHANGED: Generate "Far View" (Wide Establishing Shot) as the single source of truth.
                     const angles = [
                        { suffix: 'far', label: 'Far View', promptAdd: 'Extreme wide establishing shot. Far distance view showing the entire room/location layout from a distance. Capture the full scale, atmosphere, and furniture placement.' }
                     ];

                     const generatedLinks: string[] = [];

                     for (const angle of angles) {
                         const angleFilename = `${safeName}_env_${angle.suffix}.png`;
                         const angleRelPath = path.join('environments', angleFilename);
                         const angleAbsPath = path.join(envsDir, angleFilename);
                         
                         // Check if this specific angle exists
                         const existingRes = FileHandler.findInputFile(angleAbsPath);
                         
                         if (existingRes.found) {
                             if (!loadedGlobalImagePaths.has(existingRes.filePath!)) {
                                 try {
                                     const b64 = await FileHandler.readImageAsBase64(existingRes.filePath!);
                                     globalReferenceImages.push({ data: b64, mimeType: 'image/png', sourcePath: existingRes.filePath! });
                                     loadedGlobalImagePaths.add(existingRes.filePath!);
                                     
                                     console.error(`DEBUG - Loaded existing environment (${angle.label}): ${envName}`);
                                 } catch (e) {}
                             }
                             // Collect link for text update
                             generatedLinks.push(`![${envName} ${angle.label}](${angleRelPath})`);
                             continue;
                         }

                         // Generate Environment Image for this Angle
                         console.error(`DEBUG - Generating Environment (${angle.label}): ${envName}...`);
                         
                                         let envPrompt = `Environment Design: ${envName}. ${envDesc}.
                                 ${angle.promptAdd}
                                 ${request.style || 'shonen'} manga style, black and white background art, detailed, high quality.
                                 NO CHARACTERS. Scenery only.
                                 Establish the layout, furniture, and atmosphere described.
                                 ${request.color ? 'Full color.' : 'Black and white, screentones.'}`;
                                         
                                         // Save Environment Prompt
                                         try {
                                             const safeEnvName = envName.toLowerCase().replace(/[^a-z0-9]/g, '_');
                                             const promptFile = path.join(promptsDir, `env_${safeEnvName}_${angle.suffix}.txt`);
                                             await FileHandler.saveTextFile(promptFile, envPrompt);
                                         } catch (e) { console.error('DEBUG - Failed to save env prompt:', e); }
                         
                                         const parts: any[] = [{ text: envPrompt }];
                         try {
                            const envResponse = await this.ai.models.generateContent({
                                model: this.modelName,
                                config: {
                                  responseModalities: request.includeText ? ['IMAGE', 'TEXT'] : ['IMAGE'],
                                  imageConfig: { aspectRatio: '16:9' }, // Far view works best in wide format
                                  safetySettings: this.getSafetySettings(),
                                } as any,
                                contents: [{ role: 'user', parts: parts }],
                            });

                            if (envResponse.candidates && envResponse.candidates[0]?.content?.parts) {
                                for (const part of envResponse.candidates[0].content.parts) {
                                    let b64: string | undefined;
                                    if (part.inlineData?.data) b64 = part.inlineData.data;
                                    else if (part.text && this.isValidBase64ImageData(part.text)) b64 = part.text;

                                    if (b64) {
                                        const fullPath = await FileHandler.saveImageFromBase64(b64, envsDir, angleFilename);
                                        generatedFiles.push(fullPath);
                                        
                                        globalReferenceImages.push({ data: b64, mimeType: 'image/png', sourcePath: fullPath });
                                        loadedGlobalImagePaths.add(fullPath);
                                        
                                        generatedLinks.push(`![${envName} ${angle.label}](${angleRelPath})`);
                                        
                                        await this.logGeneration(this.modelName, [fullPath], `Environment: ${envName} (${angle.label})`);
                                        console.error(`DEBUG - Generated Environment (${angle.label}): ${fullPath}`);
                                        break;
                                    }
                                }
                            }
                         } catch (e) {
                             console.error(`DEBUG - Failed to generate environment ${envName} (${angle.label}):`, e);
                         }
                     }

                     // Update Story File with ALL links
                     // We check if the line already has these links to avoid duplication
                     let linksToAdd = "";
                     for (const link of generatedLinks) {
                         // Check if this specific link filename is already in the line
                         const filename = link.match(/\((.*?)\)/)?.[1]; // extract path inside ()
                         if (filename && !fullLine.includes(filename)) {
                             linksToAdd += ` ${link}`;
                         }
                     }

                     if (linksToAdd) {
                         newStoryContent = newStoryContent.replace(fullLine, `${fullLine}${linksToAdd}`);
                         envStoryContentModified = true;
                     }
                 }
            }
            
            if (envStoryContentModified) {
                await FileHandler.saveTextFile(storyAbsPath, newStoryContent);
                console.error(`DEBUG - Updated story file with environment links.`);
                globalContext += "\n(Note: Environment references have been auto-generated and attached)";
            }
            
            if (request.environmentGenerationOnly) {
                 const successMsg = `Successfully generated/verified environments for story. Skipping page generation.`;
                 console.error(`DEBUG - ${successMsg}`);
                 return {
                     success: true,
                     message: successMsg,
                     generatedFiles
                 };
            }
        }

        
        // 2. Fallback: Single Story Character (removed to prevent generic fallback)
        // We rely on LLM extraction or manual character references.

      // Explicit Reference Page (CLI Argument)


      }

      // Explicit Reference Page (CLI Argument)
      if (request.referencePage) {
        const refPageNum = request.referencePage.replace(/[^0-9]/g, '');
        // Try to find the file for "Page X"
        // We assume the standard naming convention: manga_page_X...
        const rawRefPrompt = `manga Page ${refPageNum}`;
        const refBaseName = rawRefPrompt.toLowerCase()
                                .replace(/[^a-z0-9\s]/g, '')
                                .replace(/\s+/g, '_')
                                .substring(0, 32);
        
        const existingRefFile = FileHandler.findLatestFile(refBaseName);
        
        if (existingRefFile) {
            try {
                const b64 = await FileHandler.readImageAsBase64(existingRefFile);
                globalReferenceImages.push({ data: b64, mimeType: 'image/png', sourcePath: existingRefFile });
                globalContext += `\n\n(See attached reference image: Page ${refPageNum}). Use this image as a strong reference for visual style and character consistency.`;
                console.error(`DEBUG - Loaded explicit reference page: ${existingRefFile}`);
            } catch (e) {
                console.error(`DEBUG - Failed to load reference page file:`, e);
            }
        } else {
            console.error(`DEBUG - Requested reference page ${request.referencePage} (file: ${refBaseName}*) not found.`);
        }
      }

      // Filter pages if specific page requested
      let pagesToProcess = pages;
      if (request.page) {
        // Split by comma or "and" to support "1, 2" or "1 and 2"
        const targets = request.page
          .split(/[,&]|\s+and\s+/)
          .map((s) => s.trim().toLowerCase())
          .filter((s) => s.length > 0);

        pagesToProcess = pages.filter((p) => {
          const headerLower = p.header.toLowerCase();
          const headerNumbers = headerLower.match(/\d+/g) || [];

          // Check if this page matches ANY of the targets
          return targets.some((target) => {
            const isTargetNumeric = /^\d+$/.test(target);
            if (isTargetNumeric) {
              // Check if ANY of the numbers in the header exactly match the target
              return headerNumbers.some((n) => n === target);
            } else {
              // Text matching
              return headerLower.includes(target);
            }
          });
        });

        if (pagesToProcess.length === 0) {
          return {
            success: false,
            message: `Page(s) "${request.page}" not found in story file.`,
            error: 'Page not found',
          };
        }
        console.error(
          `DEBUG - Filtering for pages "${request.page}". Found ${pagesToProcess.length} match(es).`,
        );
      } else if (request.startPage) {
        const target = request.startPage.trim().toLowerCase();
        const isTargetNumeric = /^\d+$/.test(target);

        const startIndex = pages.findIndex((p) => {
          const headerLower = p.header.toLowerCase();
          const headerNumbers = headerLower.match(/\d+/g) || [];
          if (isTargetNumeric) {
            return headerNumbers.some((n) => n === target);
          } else {
            return headerLower.includes(target);
          }
        });

        if (startIndex !== -1) {
          pagesToProcess = pages.slice(startIndex);
          console.error(
            `DEBUG - Starting generation from "${request.startPage}" (index ${startIndex}). Processing ${pagesToProcess.length} pages.`,
          );
        } else {
          return {
            success: false,
            message: `Start page "${request.startPage}" not found in story file.`,
            error: 'Start page not found',
          };
        }
      }

      // Iterate through pages
      for (let i = 0; i < pagesToProcess.length; i++) {
        const page = pagesToProcess[i];
        
        // Find original index to locate previous page in the full story
        const originalIndex = pages.findIndex(p => p === page);

        console.error(`DEBUG - Processing ${page.header}...`);

        // MEMORY CHECK
        const memory = await MemoryHandler.checkMemory(memoryPath, page.header);
        
        let existingArtPath: string | null = null;
        let latestFile: string | null = null;

        // Check Phase 2 (Final) Memory
        if (!request.page && memory.phase2) {
             const msg = `✅ Memory: Found PASSED Phase 2 file: ${memory.phase2}. SKIPPING generation.`;
             console.error(`DEBUG - ${msg}`);
             await this.logToDisk(msg);
             previousPagePath = memory.phase2;
             generatedFiles.push(memory.phase2);
             continue; // Skip this page
        }

        // Check Phase 1 Memory
        let phase1Prompt: string | undefined = undefined;
        if (request.twoPhase && memory.phase1) {
             existingArtPath = memory.phase1;
             phase1Prompt = memory.phase1Prompt;
             const msg = `✅ Memory: Found PASSED Phase 1 file: ${memory.phase1}. Resuming from Phase 2.`;
             console.error(`DEBUG - ${msg}`);
             await this.logToDisk(msg);
             
             if (memory.phase1PromptPath) {
                 const pMsg = `Using Phase 1 Prompt from: ${path.basename(memory.phase1PromptPath)}`;
                 console.error(`DEBUG - ${pMsg}`);
                 await this.logToDisk(pMsg);
             } else {
                 const pMsg = "Phase 1 Prompt file not found in memory. (Regenerate Phase 1 to fix this)";
                 console.error(`DEBUG - ${pMsg}`);
                 await this.logToDisk(pMsg);
             }
        }

        // Check if page already exists to support resume capability
        const filenamePrompt = `manga ${page.header}`;
        const baseName = FileHandler.getSanitizedBaseName(filenamePrompt);
        
        if (!existingArtPath) {
             console.error(`DEBUG - Checking for existing files for: "${page.header}" (Base: ${baseName})`);
             latestFile = FileHandler.findLatestFile(baseName);
        }

        // Fallback: If exact match failed, try fuzzy search by page number
        if (!latestFile && !existingArtPath) {
            const pageNumMatch = page.header.match(/Page\s*(\d+)/i);
            if (pageNumMatch) {
                console.error(`DEBUG - Exact match not found. Trying fuzzy search for Page ${pageNumMatch[1]}...`);
                latestFile = FileHandler.findPageFile(pageNumMatch[1]);
                if (latestFile) {
                    const msg = `Fuzzy match found for "${page.header}": ${latestFile}`;
                    console.error(`DEBUG - ${msg}`);
                    await this.logToDisk(msg);
                }
            }
        }

        // Only skip if NOT explicitly requested via --page
        if (!request.page && latestFile && latestFile.includes('_final')) {
            const msg = `✅ Final file found on disk: ${latestFile}. SKIPPING generation (Resume Mode).`;
            console.error(`DEBUG - ${msg}`);
            await this.logToDisk(msg);
            previousPagePath = latestFile;
            generatedFiles.push(latestFile);
            continue;
        } else if (!request.page && latestFile) {
             const msg = `Found intermediate file: ${latestFile}. Checking for resume opportunities.`;
             console.error(`DEBUG - ${msg}`);
             await this.logToDisk(msg);
        } else {
             if (!existingArtPath) {
                 const msg = `No finished file found for "${page.header}". Proceeding with generation.`;
                 console.error(`DEBUG - ${msg}`);
                 await this.logToDisk(msg);
             }
        }

        // File-based Phase 1 Art Resume Check (Fallback if memory didn't catch it)
        if (!existingArtPath && request.twoPhase && !request.page && latestFile && latestFile.includes('_phase_1')) {
            existingArtPath = latestFile;
            const msg = `Found existing Phase 1 art on disk: ${existingArtPath}. Resuming from Phase 2.`;
            console.error(`DEBUG - ${msg}`);
            await this.logToDisk(msg);
        }

        // Resolve Previous Page Reference (Logic for Continuity)
        // If we don't have a previousPagePath from this run, try to find one from disk
        if (!previousPagePath && originalIndex > 0) {
            const previousPageHeader = pages[originalIndex - 1].header;
            
            // Replicate FileHandler.generateFilename logic to find the expected filename
            const rawPrevPrompt = `manga ${previousPageHeader}`;
            const prevBaseName = rawPrevPrompt.toLowerCase()
                                    .replace(/[^a-z0-9\s]/g, '')
                                    .replace(/\s+/g, '_')
                                    .substring(0, 32);
            
            const existingPrevFile = FileHandler.findLatestFile(prevBaseName);
            
            if (existingPrevFile) {
                previousPagePath = existingPrevFile;
                console.error(`DEBUG - Found existing previous page on disk: ${existingPrevFile}`);
                console.error(`DEBUG - Using it as reference for ${page.header}`);
            } else {
                console.error(`DEBUG - No existing previous page found for ${previousPageHeader} (searched for ${prevBaseName}*)`);
            }
        }

        // Construct Prompt
        let fullPrompt = `${request.prompt}, ${ratioInstruction}\n\n[GLOBAL CONTEXT]\n${globalContext}\n\n[CURRENT PAGE: ${page.header}]\n${page.content}`;
        
        // Inject Failure Memory
        const failureData = await MemoryHandler.getFailures(memoryPath, page.header);
        if (failureData.reasons.length > 0) {
            // Filter out obsolete color warnings if we are in Two-Phase mode (where Phase 1 can be B&W)
            const activeReasons = failureData.reasons.filter(r => {
                if (request.twoPhase) {
                    const rLower = r.toLowerCase();
                    if (
                        rLower.includes("target format: full color") || 
                        rLower.includes("lack of color") || 
                        rLower.includes("black and white") ||
                        rLower.includes("lettering") || 
                        rLower.includes("typo")
                    ) {
                        return false; // Ignore old color/text failures for Phase 1
                    }
                }
                return true;
            });

            if (activeReasons.length > 0) {
                fullPrompt += `\n\n[CRITICAL WARNING: PAST FAILURES]\nThis page has failed previously. You MUST avoid the following errors:\n`;
                activeReasons.forEach(f => fullPrompt += `- ${f}\n`);
                fullPrompt += `PAY EXTRA ATTENTION TO THESE SPECIFIC ISSUES.`;
                console.error(`DEBUG - Injected ${activeReasons.length} past failure(s) into prompt.`);
            }
        }
        
        // Use failed Phase 2 image as "Layout Reference" if available
        let failedAttemptRef: { data: string, mimeType: string } | null = null;
        if (failureData.failedPaths.length > 0) {
             const lastFailedPath = failureData.failedPaths[failureData.failedPaths.length - 1];
             try {
                 const failedB64 = await FileHandler.readImageAsBase64(lastFailedPath);
                 failedAttemptRef = { data: failedB64, mimeType: 'image/png' };
                 fullPrompt += `\n\n[PREVIOUS ATTEMPT REFERENCE]\nSee attached "Reference: Previous Attempt". The text was incorrect/gibberish, but the BUBBLE LAYOUT might be useful. You may use it as a layout guide but MUST write correct text.`;
                 console.error(`DEBUG - Added failed attempt as reference: ${lastFailedPath}`);
             } catch (e) {
                 console.error(`DEBUG - Failed to load failed reference:`, e);
             }
        }

        fullPrompt += `
\n[INSTRUCTION]
Use the attached images as strict visual references.
1. **Characters**: **STRICTLY COPY** the facial features and hairstyle from the attached reference images.
   - The attached reference images are the **SUPREME AUTHORITY** and **GROUND TRUTH** for the character's **FACE AND HAIR**. You must generate the **SAME PERSON**.
   - **FACE CONSISTENCY IS MANDATORY**: Pay attention to **Eye Shape**, **Jawline**, **Nose Shape**, **Hair Parting**, and **Hair Volume**. Do not apply a generic "anime face" that erases these specific features.
   - If the character's face in the "Previous Page Reference" differs even slightly from the "Character Sheet", you MUST ignore the previous page and follow the "Character Sheet" exactly.
   - **COSTUME & CLOTHING**: 
     - IF the text in the "GLOBAL CONTEXT" (character description) or "CURRENT PAGE" describes a specific outfit (e.g., 'wearing a tuxedo' or 'battle armor'), the **TEXT OVERRIDES THE IMAGE**. Use the text for the clothing, and the reference image only for the face/hair likeness.
     - IF NO specific outfit is mentioned in the text, you MUST follow the clothing shown in the Character Sheet/Reference Image exactly.
   - **DO NOT** create a generic face. **DO NOT** hallucinate new features. Look at the "Reference Image" labeled with the character's name and **COPY IT** pixel-for-pixel where possible.
   - **ALWAYS REFER TO THE "GLOBAL CONTEXT" (Story File)** for character descriptions and details.
   - If a character appears who was NOT in the "Previous Page Reference", you MUST check the attached "Global References".
   - **DO NOT GENERATE RANDOM CHARACTERS**. If a character name matches a reference image, use that reference strictly.
   - If a character's design was established in a previous page, you must infer their consistent look from the story context provided, but the Character Sheet always overrides everything regarding physical identity.
2. **Environments**: The attached "Far View" image is your STRICT VISUAL ANCHOR. Use it to establish the room's layout, furniture placement, and atmosphere. Maintain this location's design exactly.
3. **Continuity**: If a "Previous Page reference" is attached, you MUST ensure seamless continuity. The placement of objects and characters must logically follow the previous panel. Do not teleport furniture.
4. **Text & Bubbles**: ${request.twoPhase ? 'DO NOT generate any round speech bubbles. (Rectangular captions, sound effects, and background text are PERMITTED. Only dialogue speech bubbles are forbidden.)' : `Do NOT render the page title ("${page.header.replace(/^[#\s]+/, '')}") as text in the image. You MAY render narrative captions if they are explicitly part of the panel description (e.g. "Caption: ..."), but never the page header itself.`}`;
        
        if (request.color && !request.twoPhase) {
            fullPrompt += "\n\n[STRICT COLOR MANDATE]\nGENERATE THIS PAGE IN FULL COLOR. You MUST match the EXACT color palette (hair, skin tone, eyes, clothing) from the attached Reference Images. Do not shift hues or saturation. IGNORE ANY PREVIOUS 'BLACK AND WHITE' INSTRUCTIONS.";
        } else if (request.twoPhase) {
            fullPrompt += "\n\n[STYLE MANDATE: ART PHASE]\nGENERATE THIS PAGE IN BLACK AND WHITE. Use professional manga line art, screentones, and traditional shading. NO COLOR.";
        }

        if (request.twoPhase) {
            fullPrompt += `
\n[TWO-PHASE GENERATION: ART PHASE]
IMPORTANT: This is the ART PHASE. You must generate the panels and art but **STRICTLY PROHIBITED: NO ROUND SPEECH BUBBLES**. 
- The entire frame must be filled with character and environment art.
- NOTE: Rectangular caption boxes, sound effects (SFX), and incidental text on artifacts ARE ALLOWED.
- Dialogue lines in the script should be used ONLY for determining character expressions, poses, and actions. DO NOT render the dialogue text itself inside a speech bubble.
- ZERO round white space reserved for dialogue.
- Focus entirely on character likeness, composition, and environment.
- The panels should be clean, professional illustration only.`;
        }

        // Prepare Message Parts
        const parts: any[] = [];
        
        // 1. Build Reference Mapping Table for the Prompt
        let referenceTags = "[STRICT REFERENCE MAPPING]\nThe following images are attached. You MUST use the specific image when the corresponding Name/Tag appears in the story:\n";
        
        // Global Refs
        for (const img of globalReferenceImages) {
            const label = path.basename(img.sourcePath, path.extname(img.sourcePath)).replace(/_/g, ' ');
            // Simple heuristic: "kenji_portrait" -> Tag: "kenji"
            // "unity_hq_far" -> Tag: "unity hq"
            const tag = label.replace(/\s+(portrait|sheet|reference|ref|far|view|env|environment)$/i, '').trim();
            referenceTags += `- Tag: "${tag}" matches Reference Image: "${label}"\n`;
        }

        // Page Refs
        const pageImagePaths = this.extractImagePaths(page.content);
        for (const imgPath of pageImagePaths) {
            if (globalImagePaths.includes(imgPath)) continue;
             const fileRes = FileHandler.findInputFile(imgPath);
             if (fileRes.found) {
                const label = path.basename(fileRes.filePath!, path.extname(fileRes.filePath!)).replace(/_/g, ' ');
                const tag = label.replace(/\s+(portrait|sheet|reference|ref|far|view|env|environment)$/i, '').trim();
                referenceTags += `- Tag: "${tag}" matches Reference Image: "${label}"\n`;
             }
        }
        
        referenceTags += "\n[INSTRUCTION]\nWhen you see a Tag in the text (e.g., 'Kenji enters'), look up the corresponding 'Reference Image' above and Apply it STRICTLY.\n";

        // Append to fullPrompt
        fullPrompt += `\n\n${referenceTags}`;

        parts.push({ text: fullPrompt });

        // Add Failed Attempt Reference
        if (failedAttemptRef) {
             parts.push({ text: "Reference: Previous Attempt (Bad Text/Layout)" });
             parts.push({ inlineData: failedAttemptRef });
        }

        // Add Global References with Labels
        for (const img of globalReferenceImages) {
            const label = path.basename(img.sourcePath, path.extname(img.sourcePath)).replace(/_/g, ' ');
            parts.push({ text: `Reference image for: "${label}"` });
            parts.push({ inlineData: { data: img.data, mimeType: img.mimeType } });
        }

        // Add Page Specific References with Labels
        for (const imgPath of pageImagePaths) {
            // Avoid duplicates if already in global
            if (globalImagePaths.includes(imgPath)) continue;

            const fileRes = FileHandler.findInputFile(imgPath);
            if (fileRes.found) {
                try {
                    const b64 = await FileHandler.readImageAsBase64(fileRes.filePath!);
                    const label = path.basename(fileRes.filePath!, path.extname(fileRes.filePath!)).replace(/_/g, ' ');
                    parts.push({ text: `Reference image for: "${label}"` });
                    parts.push({ inlineData: { data: b64, mimeType: 'image/png' } });
                    console.error(`DEBUG - Loaded page reference: ${imgPath}`);
                } catch (e) {
                    console.error(`DEBUG - Failed to load page ref ${imgPath}:`, e);
                }
            }
        }

        // Add Previous Page Reference (Sequential Consistency)
        if (previousPagePath) {
            try {
                const prevB64 = await FileHandler.readImageAsBase64(previousPagePath);
                parts.push({ text: "Reference: Previous Page" });
                parts.push({ inlineData: { data: prevB64, mimeType: 'image/png' } });
                parts[0].text += `\n\n[PREVIOUS PAGE REFERENCE]\nThe attached image "Reference: Previous Page" is the immediately preceding page (${originalIndex > 0 ? pages[originalIndex - 1].header : 'previous'}). Maintain strict visual continuity with it regarding environment, lighting, and character positioning where applicable.`;
                console.error(`DEBUG - Added previous page as reference.`);
            } catch (e) {
                console.error(`DEBUG - Failed to load previous page ref:`, e);
            }
        }

        // LOGGING INCLUDED IMAGES
        try {
            const includedImages: string[] = [];
            
            // Global Refs
            globalReferenceImages.forEach(img => includedImages.push(`Global Ref: ${path.basename(img.sourcePath)}`));
            
            // Page Refs
            for (const imgPath of pageImagePaths) {
                 if (!globalImagePaths.includes(imgPath)) {
                      const fileRes = FileHandler.findInputFile(imgPath);
                      if (fileRes.found) includedImages.push(`Page Ref: ${path.basename(fileRes.filePath!)}`);
                 }
            }

            // Previous Page
            if (previousPagePath) includedImages.push(`Prev Page Ref: ${path.basename(previousPagePath)}`);

            const logDir = FileHandler.ensureOutputDirectory();
            const logFile = path.join(logDir, 'nanobanana-output.log');
            const timestamp = new Date().toISOString();
            const aspectRatio = this.getAspectRatioString(request.layout);
            const logEntry = `Generating ${page.header}. Aspect Ratio: ${aspectRatio}. Attached References: ${includedImages.join(', ')}`;
            
            await this.logToDisk(logEntry);
            console.error(`DEBUG - Logged attached references to ${logFile}`);
            console.error(`DEBUG - Generating with Aspect Ratio: ${aspectRatio}`);

        } catch (e) {
            console.error('DEBUG - Failed to log attached references:', e);
        }

        // RETRY LOOP FOR GENERATION
        let attemptSuccess = false;
        let finalGeneratedPath = "";
        const maxRetries = request.retryCount || 3;
        
        // Store original prompt to append corrections without duplication
        const originalPromptText = parts[0].text;
        let correctionInstruction = "";

        for (let attempt = 1; attempt <= maxRetries; attempt++) {
            let fullPath = "";
            let imageSaved = false;

            // Phase 1 Resume Logic: Use existing art if available (even on retries)
            if (existingArtPath) {
                fullPath = existingArtPath;
                imageSaved = true;
                const msg = `Resuming ${page.header} using existing Phase 1 art: ${existingArtPath}`;
                console.error(`DEBUG - ${msg}`);
                await this.logToDisk(msg);
            } else {
                console.error(`DEBUG - Generating ${page.header} (Attempt ${attempt}/${maxRetries})...`);
                
                // Apply correction instruction if exists
                if (correctionInstruction) {
                    parts[0].text = originalPromptText + correctionInstruction;
                    console.error(`DEBUG - Applied correction instruction to prompt.`);
                }
                
                // Save Phase 1 Prompt (Before Generation)
                try {
                    const safeHeader = FileHandler.getSanitizedBaseName(page.header);
                    const promptFile = path.join(promptsDir, `page_${safeHeader}_phase1_attempt${attempt}.txt`);
                    await FileHandler.saveTextFile(promptFile, parts[0].text);
                    const logMsg = `Saved Phase 1 Prompt to: ${path.basename(promptFile)}`;
                    console.error(`DEBUG - ${logMsg}`);
                    await this.logToDisk(logMsg);
                } catch (e) { console.error('DEBUG - Failed to save Phase 1 prompt:', e); }

                try {
                    const response = await this.ai.models.generateContent({
                      model: this.artModel,
                      contents: [{ role: 'user', parts: parts }],
                      config: {
                        responseModalities: (request.twoPhase ? false : request.includeText) ? ['IMAGE', 'TEXT'] : ['IMAGE'],
                        imageConfig: {
                          aspectRatio: this.getAspectRatioString(request.layout),
                        },
                        safetySettings: this.getSafetySettings(),
                      } as any,
                    });
            
                    if (response.candidates && response.candidates[0]?.content?.parts) {
                      for (const part of response.candidates[0].content.parts) {
                        let imageBase64: string | undefined;
                        if (part.inlineData?.data) {
                          imageBase64 = part.inlineData.data;
                        } else if (part.text && this.isValidBase64ImageData(part.text)) {
                          imageBase64 = part.text;
                        }
            
                        if (imageBase64) {
                          // Save with _phase_1 suffix if in two-phase mode
                          const phase1Suffix = request.twoPhase ? ' phase 1' : '';
                          const filename = FileHandler.generateFilename(
                            `manga ${page.header}${phase1Suffix}`,
                            'png',
                            0,
                          );
                          fullPath = await FileHandler.saveImageFromBase64(
                            imageBase64,
                            FileHandler.ensureOutputDirectory(),
                            filename,
                          );
                          
                          await this.logGeneration(this.artModel, [fullPath], `Phase 1 (Art) for ${page.header}`);
                          imageSaved = true;
                          break;
                        }
                      }
                    }
                } catch (error: unknown) {
                    const msg = this.handleApiError(error);
                    console.error(`DEBUG - Error generating ${page.header} (Attempt ${attempt}):`, msg);
                    if (attempt === maxRetries && !firstError) firstError = msg;
                    continue; // Try next attempt
                }
            }

            if (imageSaved && fullPath) {
                try {
                      let phase1Warning = "";

                      // Phase 1 Review (Likeness, Continuity, NO BUBBLES)
                      if (request.twoPhase) {
                          const contextForReview = `Scene Prompt: ${request.prompt || ''}\nScript Content: ${page.content}`;
                          const reviewRefs = [...globalReferenceImages];
                          if (previousPagePath) {
                              try {
                                 const prevB64 = await FileHandler.readImageAsBase64(previousPagePath);
                                 reviewRefs.push({ data: prevB64, mimeType: 'image/png', sourcePath: 'Previous Page Reference' });
                              } catch (e) {}
                          }

                          const phase1Review = await this.reviewGeneratedImage(fullPath, reviewRefs, {
                              minScore: request.minScore || 8,
                              minLikeness: request.minLikeness,
                              minStory: request.minStory,
                              minContinuity: request.minContinuity,
                              minLettering: request.minLettering,
                              minNoBubbles: request.minNoBubbles,
                              storyContext: contextForReview,
                              isPhase1: true,
                              isColor: request.color
                          });

                          if (!phase1Review.pass) {
                              const errorMsg = `Phase 1 Review FAILED for ${page.header} (Attempt ${attempt}). Score: ${phase1Review.score}/400. Reason: ${phase1Review.reason}.`;
                              console.error(`❌ ${errorMsg}`);
                              await this.logToDisk(errorMsg);
                              
                              // Log Failure to Memory
                              await MemoryHandler.updateMemory(memoryPath, page.header, 1, 'FAILED', { reason: phase1Review.reason });
                              
                              // Delete failed file
                              try {
                                  await fs.promises.unlink(fullPath);
                                  console.error(`DEBUG - Deleted failed Art image: ${fullPath}`);
                              } catch (e) {
                                  console.error(`DEBUG - Failed to delete ${fullPath}:`, e);
                              }
                              
                              // Clear existingArtPath to ensure next attempt actually generates
                              existingArtPath = null;
                              
                              // Add correction instruction
                              correctionInstruction = `\n\n[CRITICAL ART CORRECTION]\nPrevious Art rejected: ${phase1Review.reason}\nFix likeness and ENSURE NO ROUND SPEECH BUBBLES. (Rectangular captions, SFX, and background text are okay).`;
                              
                              if (attempt === maxRetries) {
                                  return { success: false, message: `Failed at Phase 1 after ${maxRetries} attempts.`, error: errorMsg, generatedFiles };
                              }
                              continue; // Retry Phase 1
                          } else {
                              // If passed but has warnings (especially bubbles), capture them
                              if (phase1Review.reason && (phase1Review.reason.toLowerCase().includes('bubble') || phase1Review.reason.toLowerCase().includes('text'))) {
                                  phase1Warning = phase1Review.reason;
                                  console.error(`DEBUG - Phase 1 Warning Captured: ${phase1Warning}`);
                              }
                          }
                          const msg = `✅ Phase 1 Passed Review. Proceeding to Phase 2.`;
                                            console.error(msg);
                                            await this.logToDisk(msg);
                                            
                                                                                // Update Memory
                                            
                                                                                await MemoryHandler.updateMemory(memoryPath, page.header, 1, 'PASSED', { filePath: fullPath, prompt: originalPromptText });
                                            
                                                                              }
                                            
                                                              
                                            
                                                                              let finalPathForReview = fullPath;                      // Phase 2: Add Text & Color
                      if (request.twoPhase) {
                          try {
                              // Collect all relevant references for Phase 2 (Colorization)
                              const phase2Refs = [...globalReferenceImages];
                              for (const imgPath of pageImagePaths) {
                                  if (!globalImagePaths.includes(imgPath)) {
                                      const fileRes = FileHandler.findInputFile(imgPath);
                                      if (fileRes.found) {
                                          const b64 = await FileHandler.readImageAsBase64(fileRes.filePath!);
                                          phase2Refs.push({ data: b64, mimeType: 'image/png', sourcePath: fileRes.filePath! });
                                      }
                                  }
                              }

                              finalPathForReview = await this.addTextToMangaPage(
                                  fullPath, 
                                  page.content, 
                                  page.header,
                                  phase2Refs,
                                  request.color || false,
                                  phase1Prompt || originalPromptText,
                                  phase1Warning,
                                  promptsDir
                              );
                          } catch (e) {
                              console.error(`DEBUG - Phase 2 failed, falling back to Phase Art image for review.`, e);
                          }
                      }

                      // Auto-Review Step
                      const contextForReview = `Scene Prompt: ${request.prompt || ''}\nScript Content: ${page.content}`;
                      
                      // Prepare references including Previous Page for consistency check
                      const reviewRefs = [...globalReferenceImages];
                      if (previousPagePath) {
                          try {
                             const prevB64 = await FileHandler.readImageAsBase64(previousPagePath);
                             reviewRefs.push({
                                 data: prevB64,
                                 mimeType: 'image/png',
                                 sourcePath: 'Previous Page Reference'
                             });
                          } catch (e) {
                              console.error(`DEBUG - Failed to load previous page for review:`, e);
                          }
                      }

                      const review = await this.reviewGeneratedImage(finalPathForReview, reviewRefs, {
                        minScore: request.minScore || 8,
                        minLikeness: request.minLikeness,
                        minStory: request.minStory,
                        minContinuity: request.minContinuity,
                        minLettering: request.minLettering,
                        storyContext: contextForReview
                      });
                      
                      if (review.pass) {
                          generatedFiles.push(finalPathForReview);
                          await this.logGeneration(request.twoPhase ? this.textModel : this.artModel, [finalPathForReview]);
                          
                          // Cleanup intermediate Phase 1 Art file
                          if (request.twoPhase && finalPathForReview !== fullPath) {
                              try {
                                  await fs.promises.unlink(fullPath);
                                  console.error(`DEBUG - Cleaned up intermediate Art file: ${fullPath}`);
                              } catch (e) {
                                  console.error(`DEBUG - Failed to delete intermediate file:`, e);
                              }
                          }

                          previousPagePath = finalPathForReview; // Update for next iteration
                          finalGeneratedPath = finalPathForReview;
                          attemptSuccess = true;
                                                    const msg = `SUCCESS: Generated ${page.header} on attempt ${attempt}. Score: ${review.score}`;
                                                    console.error(`DEBUG - ${msg}`);
                                                    await this.logToDisk(msg);
                                                    
                                                                                                        // Update Memory
                                                    
                                                                                                        await MemoryHandler.updateMemory(memoryPath, page.header, 2, 'PASSED', { filePath: finalPathForReview });
                                                    
                                                                                                        break;
                                                    
                                                                                                    } else {                          const errorMsg = `Review FAILED for ${page.header} (Attempt ${attempt}). Score: ${review.score}/10. Reason: ${review.reason}.`;
                          console.error(`❌ ${errorMsg}`);
                          await this.logToDisk(errorMsg);
                          
                          // Log Failure to Memory
                          await MemoryHandler.updateMemory(memoryPath, page.header, 2, 'FAILED', { reason: review.reason });
                          
                          // Delete failed files
                          try {
                              if (finalPathForReview !== fullPath) {
                                  // RENAME failed Phase 2 file for reference instead of deleting
                                  const dir = path.dirname(finalPathForReview);
                                  const ext = path.extname(finalPathForReview);
                                  const base = path.basename(finalPathForReview, ext);
                                  const failedFilename = `${base}_failed_${Date.now()}${ext}`;
                                  const failedPath = path.join(dir, failedFilename);
                                  
                                  await fs.promises.rename(finalPathForReview, failedPath);
                                  console.error(`DEBUG - Renamed failed Final image to: ${failedPath}`);
                                  
                                  // Log Failure to Memory with path
                                  await MemoryHandler.updateMemory(memoryPath, page.header, 2, 'FAILED', { reason: review.reason, failedPath: failedPath });
                              } else {
                                   await MemoryHandler.updateMemory(memoryPath, page.header, 2, 'FAILED', { reason: review.reason });
                              }
                              
                              if (request.twoPhase) {
                                  // Preserve Phase 1 Art for retry
                                  console.error(`DEBUG - Preserving Phase 1 Art for retry: ${fullPath}`);
                                  existingArtPath = fullPath;
                              } else {
                                  // Single phase or fallback - delete if failed
                                  if (fullPath !== finalPathForReview || !request.twoPhase) {
                                       await fs.promises.unlink(fullPath);
                                       console.error(`DEBUG - Deleted failed image: ${fullPath}`);
                                  }
                                  existingArtPath = null;
                              }
                          } catch (e) {
                              console.error(`DEBUG - Failed to clean up failed images:`, e);
                          }
                          
                          // Dynamic Correction Logic
                          const reasonLower = review.reason.toLowerCase();
                          let specificFix = "";

                          if (reasonLower.includes('face') || reasonLower.includes('facial') || reasonLower.includes('eyes') || reasonLower.includes('likeness') || reasonLower.includes('look like') || reasonLower.includes('older') || reasonLower.includes('younger')) {
                              specificFix += `\n- [FACE IDENTITY FIX]: The previous face was WRONG. You must COPY the facial features from the Reference Image labeled with the character's name. MATCH THE EYE SHAPE AND JAWLINE EXACTLY. Do not stylize into a generic face.`;
                          }
                          if (reasonLower.includes('hair') || reasonLower.includes('hairstyle')) {
                              specificFix += `\n- [HAIR FIX]: The hairstyle was WRONG. Look at the Reference Image and copy the hair volume, bangs, and parting EXACTLY.`;
                          }
                          if (reasonLower.includes('style') || reasonLower.includes('rendering')) {
                              specificFix += `\n- [STYLE FIX]: The art style was inconsistent. Ensure line weight and shading match the 'Previous Page Reference'.`;
                          }
                          if (reasonLower.includes('text') || reasonLower.includes('gibberish') || reasonLower.includes('lettering') || reasonLower.includes('spelling')) {
                              specificFix += `\n- [TEXT FIX]: The previous text was incorrect or gibberish. You MUST use the EXACT text from the "Story Script". Do NOT hallucinate words. Ensure every bubble is filled with legible English text matching the script.`;
                          }

                          // Set correction instruction for next attempt
                          correctionInstruction = `\n\n[CRITICAL CORRECTION REQUIRED]\nThe previous generation was REJECTED (Score: ${review.score}/10).
**FAILURE REASON**: ${review.reason}
**INSTRUCTION**: You MUST fix the specific issue identified above.
${specificFix}
- If the environment was wrong, align with the "Far View" anchor.
- DO NOT IGNORE THIS. You will fail again if you do not correct these specific details.`;

                          if (attempt === maxRetries) {
                              return {
                                  success: false,
                                  message: `Generation stopped. Failed to generate consistent image for ${page.header} after ${maxRetries} attempts.`,
                                  error: errorMsg,
                                  generatedFiles: generatedFiles
                              };
                          }
                      }
                } catch (e: unknown) {
                    const msg = e instanceof Error ? e.message : String(e);
                    console.error(`DEBUG - Error in generation process for ${page.header}:`, msg);
                    if (attempt === maxRetries && !firstError) firstError = msg;
                }
            }
            
            if (attemptSuccess) break; // Break retry loop
        }
      }

      if (generatedFiles.length > 0) {
        await this.handlePreview(generatedFiles, request);
        return {
          success: true,
          message: `Successfully generated ${generatedFiles.length} of ${pages.length} manga pages/panels.`,
          generatedFiles,
        };
      }

      return {
        success: false,
        message: 'Failed to generate manga pages',
        error: firstError || 'No images generated',
      };
    } catch (error: unknown) {
      console.error('DEBUG - Error in generateMangaPage:', error);
      return {
        success: false,
        message: 'Failed to generate manga page',
        error: this.handleApiError(error),
      };
    }
  }

  async editImage(
    request: ImageGenerationRequest,
  ): Promise<ImageGenerationResponse> {
    try {
      if (!request.inputImage) {
        return {
          success: false,
          message: 'Input image file is required for editing',
          error: 'Missing inputImage parameter',
        };
      }

      const fileResult = FileHandler.findInputFile(request.inputImage);
      if (!fileResult.found) {
        return {
          success: false,
          message: `Input image not found: ${request.inputImage}`,
          error: `Searched in: ${fileResult.searchedPaths.join(', ')}`,
        };
      }

      const outputPath = FileHandler.ensureOutputDirectory();
      const imageBase64 = await FileHandler.readImageAsBase64(
        fileResult.filePath!,
      );

      const response = await this.ai.models.generateContent({
        model: this.modelName,
        config: {
          responseModalities: request.includeText ? ['IMAGE', 'TEXT'] : ['IMAGE'],
          safetySettings: this.getSafetySettings(),
        } as any,
        contents: [
          {
            role: 'user',
            parts: [
              { text: request.prompt },
              {
                inlineData: {
                  data: imageBase64,
                  mimeType: 'image/png',
                },
              },
            ],
          },
        ],
      });

      console.error(
        'DEBUG - Edit API Response structure:',
        JSON.stringify(response, null, 2),
      );

      if (response.candidates && response.candidates[0]?.content?.parts) {
        const generatedFiles: string[] = [];
        let imageFound = false;

        for (const part of response.candidates[0].content.parts) {
          let resultImageBase64: string | undefined;

          if (part.inlineData?.data) {
            resultImageBase64 = part.inlineData.data;
            console.error('DEBUG - Found edited image in inlineData:', {
              length: resultImageBase64?.length ?? 0,
              mimeType: part.inlineData.mimeType,
            });
          } else if (part.text && this.isValidBase64ImageData(part.text)) {
            resultImageBase64 = part.text;
            console.error(
              'DEBUG - Found edited image in text field (fallback)',
            );
          }

          if (resultImageBase64) {
            const filename = FileHandler.generateFilename(
              `${request.mode}_${request.prompt}`,
              'png', // Edits default to png
              0,
            );
            const fullPath = await FileHandler.saveImageFromBase64(
              resultImageBase64,
              outputPath,
              filename,
            );
generatedFiles.push(fullPath);
            console.error('DEBUG - Edited image saved to:', fullPath);
            imageFound = true;
            break; // Only process the first valid image
          }
        }

        if (!imageFound) {
          console.error(
            'DEBUG - No valid image data found in edit response parts',
          );
        }

        // Handle preview if requested
        await this.handlePreview(generatedFiles, request);

        await this.logGeneration(this.modelName, generatedFiles);

        return {
          success: true,
          message: `Successfully ${request.mode}d image`,
          generatedFiles,
        };
      }

      return {
        success: false,
        message: `Failed to ${request.mode} image`,
        error: 'No image data in response',
      };
    } catch (error: unknown) {
      console.error(`DEBUG - Error in ${request.mode}Image:`, error);
      return {
        success: false,
        message: `Failed to ${request.mode} image`,
        error: this.handleApiError(error),
      };
    }
  }
}

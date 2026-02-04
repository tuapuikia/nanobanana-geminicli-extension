/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { GoogleGenAI } from '@google/genai';
import { FileHandler } from './fileHandler.js';
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
  private static readonly DEFAULT_MODEL = 'gemini-2.5-flash-image';

  constructor(authConfig: AuthConfig) {
    this.ai = new GoogleGenAI({
      apiKey: authConfig.apiKey,
    });
    this.modelName =
      process.env.NANOBANANA_MODEL || ImageGenerator.DEFAULT_MODEL;
    console.error(`DEBUG - Using image model: ${this.modelName}`);
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

  private async logGeneration(modelName: string, generatedFiles: string[], referenceInfo?: string): Promise<void> {
    try {
      const logDir = FileHandler.ensureOutputDirectory();
      const logFile = path.join(logDir, 'nanobanana-output.log');
      const timestamp = new Date().toISOString();
      let logEntry = `[${timestamp}] Model: ${modelName}, Generated Files: ${generatedFiles.join(', ')}`;
      
      if (referenceInfo) {
          logEntry += `, Reference: ${referenceInfo}`;
      }
      logEntry += '\n';

      await fs.promises.appendFile(logFile, logEntry, 'utf-8');
      console.error(`DEBUG - Logged generation to: ${logFile}`);
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
                const escapedName = sourceName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
                const descRegex = new RegExp(`(?:^|\\n)\\s*[\\*\\-]?\\s*\\*?\\*?${escapedName}\\*?\\*?(?::)?\\s*([^*\\n]+)`, 'i');
                const match = storyText.match(descRegex);
                if (match) {
                    characterDescription = match[1].trim();
                    console.error(`DEBUG - Found description for ${sourceName}: ${characterDescription}`);
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

      // Parse Story Content for Pages
      // Splits by headers like "# Page 1", "## Page 2", "Page 3:"
      const pageRegex = /(?:^|\n)(#{1,3}\s*Page\s*\d+|Page\s*\d+:)/i;
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
                         
                         const bwParts: any[] = [{ text: bwPrompt }];
                         if (sourceImageB64) {
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
                     // We will prioritize the new multi-angle format: _env_top.png, _env_main.png, _env_reverse.png
                     
                     // ORDER CHANGED: Top View FIRST to establish layout, then Main (Front), then Reverse.
                     const angles = [
                        { suffix: 'top', label: 'Top View', promptAdd: 'Top-down floor plan view. Architectural layout map. Show furniture placement clearly.' },
                        { suffix: 'main', label: 'Main View', promptAdd: 'Wide establishing shot (Front View). Show the entire room layout as seen from the entrance or main angle.' },
                        { suffix: 'reverse', label: 'Reverse View', promptAdd: 'Reverse angle shot (Back View). Camera looking from the opposite direction. Show the other side of the room.' }
                     ];

                     const generatedLinks: string[] = [];
                     let topViewB64: string | undefined;

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
                                     
                                     if (angle.suffix === 'top') topViewB64 = b64;
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
                         
                         // If generating Main or Reverse view, use Top view as strict reference
                         if (topViewB64 && angle.suffix !== 'top') {
                             envPrompt += " Use the attached Top-Down Floor Plan as the STRICT ARCHITECTURAL BLUEPRINT. Align all furniture placement, walls, and doors exactly as shown in the plan.";
                         }
                         
                         const parts: any[] = [{ text: envPrompt }];
                         if (topViewB64 && angle.suffix !== 'top') {
                             parts.push({ inlineData: { data: topViewB64, mimeType: 'image/png' } });
                         }

                         try {
                            const envResponse = await this.ai.models.generateContent({
                                model: this.modelName,
                                config: {
                                  responseModalities: request.includeText ? ['IMAGE', 'TEXT'] : ['IMAGE'],
                                  imageConfig: { aspectRatio: angle.suffix === 'top' ? '1:1' : '16:9' }, // Top view is square for better map view
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
                                        
                                        if (angle.suffix === 'top') topViewB64 = b64;
                                        
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

        // Check if page already exists to support resume capability
        const filenamePrompt = `manga ${page.header}`;
        const baseName = FileHandler.getSanitizedBaseName(filenamePrompt);
        // We assume 'png' as per generateFilename usage later in the loop
        const expectedFilename = `${baseName}.png`;
        const outputPath = FileHandler.ensureOutputDirectory();
        const fullExpectedPath = path.join(outputPath, expectedFilename);

        // Only skip if NOT explicitly requested via --page
        if (!request.page && fs.existsSync(fullExpectedPath)) {
            console.error(`DEBUG - File already exists: ${fullExpectedPath}. Skipping generation (Resume Mode).`);
            previousPagePath = fullExpectedPath;
            generatedFiles.push(fullExpectedPath);
            continue;
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
        fullPrompt += `
\n[INSTRUCTION]
Use the attached images as strict visual references.
1. **Characters**: Maintain specific appearance (hairstyle, clothing, features) consistently.
2. **Environments**: If an environment reference is provided (e.g. "Living Room", "Kitchen"), you MUST maintain the exact spatial layout, furniture placement, and architectural details. Do not move fixed objects like windows, doors, or large furniture (sofas, desks) unless the script explicitly moves them.
3. **Continuity**: Ensure the visual style matches the "Previous Page" reference if provided.`;
        
        if (request.color) {
            fullPrompt += "\n\n[IMPORTANT STYLE OVERRIDE]\nGENERATE THIS PAGE IN FULL COLOR. IGNORE ANY PREVIOUS 'BLACK AND WHITE' INSTRUCTIONS.";
        }

        // Prepare Message Parts
        const parts: any[] = [{ text: fullPrompt }];

        // Add Global References with Labels
        for (const img of globalReferenceImages) {
            const label = path.basename(img.sourcePath, path.extname(img.sourcePath)).replace(/_/g, ' ');
            parts.push({ text: `Reference image for: "${label}"` });
            parts.push({ inlineData: { data: img.data, mimeType: img.mimeType } });
        }

        // Add Page Specific References with Labels
        const pageImagePaths = this.extractImagePaths(page.content);
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
            // Note: We iterated pageImagePaths earlier but didn't store source paths in a list, 
            // but we can infer them or just log what we found.
            // Let's re-check what was actually added.
            // Actually, we can just iterate pageImagePaths again since we only added if found.
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
            const logEntry = `[${timestamp}] Generating ${page.header}. Aspect Ratio: ${aspectRatio}. Attached References: ${includedImages.join(', ')}\n`;
            
            await fs.promises.appendFile(logFile, logEntry, 'utf-8');
            console.error(`DEBUG - Logged attached references to ${logFile}`);
            console.error(`DEBUG - Generating with Aspect Ratio: ${aspectRatio}`);

        } catch (e) {
            console.error('DEBUG - Failed to log attached references:', e);
        }

        try {
            const response = await this.ai.models.generateContent({
              model: this.modelName,
              contents: [{ role: 'user', parts: parts }],
              config: {
                responseModalities: request.includeText ? ['IMAGE', 'TEXT'] : ['IMAGE'],
                imageConfig: {
                  aspectRatio: this.getAspectRatioString(request.layout),
                },
                safetySettings: this.getSafetySettings(),
              } as any,
            });
    
            let imageSaved = false;
            if (response.candidates && response.candidates[0]?.content?.parts) {
              for (const part of response.candidates[0].content.parts) {
                let imageBase64: string | undefined;
                if (part.inlineData?.data) {
                  imageBase64 = part.inlineData.data;
                } else if (part.text && this.isValidBase64ImageData(part.text)) {
                  imageBase64 = part.text;
                }
    
                if (imageBase64) {
                  // Determine filename based on page header or index
                  const filenamePrompt = `manga ${page.header}`;
                  const filename = FileHandler.generateFilename(
                    filenamePrompt,
                    'png',
                    0,
                  );
                  const fullPath = await FileHandler.saveImageFromBase64(
                    imageBase64,
                    FileHandler.ensureOutputDirectory(),
                    filename,
                  );
                  generatedFiles.push(fullPath);
                  await this.logGeneration(this.modelName, [fullPath]);
                  previousPagePath = fullPath; // Update for next iteration
                  imageSaved = true;
                  console.error(`DEBUG - Generated ${page.header}: ${fullPath}`);
                  break; 
                }
              }
            }

            if (!imageSaved) {
                const msg = `Failed to generate image data for ${page.header}`;
                console.error(`DEBUG - ${msg}`);
                if (!firstError) firstError = msg;
            }

        } catch (error: unknown) {
            const msg = this.handleApiError(error);
            console.error(`DEBUG - Error generating ${page.header}:`, msg);
            if (!firstError) firstError = msg;
            // Decide whether to continue? For sequential manga, failure in middle breaks chain.
            // But maybe we should try best effort for remaining pages? 
            // Let's continue but previousPagePath will stay as the last successful one (or null).
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

/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { GoogleGenAI } from '@google/genai';
import { FileHandler } from './fileHandler.js';
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
                
                const response = await this.ai.models.generateContent({
                    model: this.modelName,
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
          const response = await this.ai.models.generateContent({
            model: this.modelName,
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
      const globalReferenceImages: { data: string; mimeType: string }[] = [];

      for (const imgPath of globalImagePaths) {
        const fileRes = FileHandler.findInputFile(imgPath);
        if (fileRes.found) {
            try {
                const b64 = await FileHandler.readImageAsBase64(fileRes.filePath!);
                globalReferenceImages.push({
                    data: b64,
                    mimeType: 'image/png' // Assuming png for simplicity, logic could be smarter
                });
                console.error(`DEBUG - Loaded global reference: ${imgPath}`);
            } catch (e) {
                console.error(`DEBUG - Failed to load global ref ${imgPath}:`, e);
            }
        }
      }

      // Explicit Character Image (CLI Argument)
      if (request.characterImage) {
        const charRes = FileHandler.findInputFile(request.characterImage);
        if (charRes.found) {
             const b64 = await FileHandler.readImageAsBase64(charRes.filePath!);
             globalReferenceImages.push({ data: b64, mimeType: 'image/png' });
             globalContext += `\n\n(See attached character reference: ${request.characterImage})`;
        }
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
                globalReferenceImages.push({ data: b64, mimeType: 'image/png' });
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
        
        if (request.color) {
            fullPrompt += "\n\n[IMPORTANT STYLE OVERRIDE]\nGENERATE THIS PAGE IN FULL COLOR. IGNORE ANY PREVIOUS 'BLACK AND WHITE' INSTRUCTIONS.";
        }

        // Prepare Message Parts
        const parts: any[] = [{ text: fullPrompt }];

        // Add Global References
        for (const img of globalReferenceImages) {
            parts.push({ inlineData: img });
        }

        // Add Page Specific References
        const pageImagePaths = this.extractImagePaths(page.content);
        for (const imgPath of pageImagePaths) {
            // Avoid duplicates if already in global
            if (globalImagePaths.includes(imgPath)) continue;

            const fileRes = FileHandler.findInputFile(imgPath);
            if (fileRes.found) {
                try {
                    const b64 = await FileHandler.readImageAsBase64(fileRes.filePath!);
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
                parts.push({ inlineData: { data: prevB64, mimeType: 'image/png' } });
                parts[0].text += `\n\n[PREVIOUS PAGE REFERENCE]\nThe attached image is the immediately preceding page (${originalIndex > 0 ? pages[originalIndex - 1].header : 'previous'}). Maintain strict visual continuity with it regarding environment, lighting, and character positioning where applicable.`;
                console.error(`DEBUG - Added previous page as reference.`);
            } catch (e) {
                console.error(`DEBUG - Failed to load previous page ref:`, e);
            }
        }

        try {
            const response = await this.ai.models.generateContent({
              model: this.modelName,
              contents: [{ role: 'user', parts: parts }],
              config: {
                aspectRatio: this.getAspectRatioString(request.layout),
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

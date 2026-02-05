/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

export interface ImageGenerationRequest {
  prompt: string;
  inputImage?: string;
  inputDirectory?: string;
  outputCount?: number;
  mode: 'generate' | 'edit' | 'restore' | 'manga';
  // Batch generation options
  styles?: string[];
  variations?: string[];
  format?: 'grid' | 'separate';
  fileFormat?: 'png' | 'jpeg';
  seed?: number;
  // Preview options
  preview?: boolean;
  noPreview?: boolean;
  // Manga specific
  storyFile?: string;
  characterImage?: string;
  referencePage?: string;
  page?: string;
  startPage?: string;
  layout?: string;
  style?: string;
  color?: boolean;
  createCharacter?: boolean;
  autoGenerateCharacters?: boolean;
  characterGenerationOnly?: boolean;
  autoGenerateEnvironments?: boolean;
  environmentGenerationOnly?: boolean;
  includeText?: boolean;
  minScore?: number;
  minLikeness?: number;
  minStory?: number;
  minContinuity?: number;
  retryCount?: number;
}

export interface ImageGenerationResponse {
  success: boolean;
  message: string;
  generatedFiles?: string[];
  error?: string;
}

export interface AuthConfig {
  apiKey: string;
  keyType: 'GEMINI_API_KEY' | 'GOOGLE_API_KEY';
}

export interface FileSearchResult {
  found: boolean;
  filePath?: string;
  searchedPaths: string[];
}

export interface StorySequenceArgs {
  type?: string;
  style?: string;
  transition?: string;
}

export interface IconPromptArgs {
  prompt?: string;
  type?: string;
  style?: string;
  background?: string;
  corners?: string;
}

export interface PatternPromptArgs {
  prompt?: string;
  type?: string;
  style?: string;
  density?: string;
  colors?: string;
  size?: string;
}

export interface DiagramPromptArgs {
  prompt?: string;
  type?: string;
  style?: string;
  layout?: string;
  complexity?: string;
  colors?: string;
  annotations?: string;
}

export interface MangaPromptArgs {
  prompt?: string;
  story_file?: string;
  input_image?: string;
  input_directory?: string;
  character_image?: string;
  reference_page?: string;
  style?: string;
  layout?: string;
  page?: string;
  from_page?: string;
  color?: boolean;
  create_character?: boolean;
  preview?: boolean;
  noPreview?: boolean;
  generate_characters?: boolean;
  character_generation_only?: boolean;
  generate_environments?: boolean;
  environment_generation_only?: boolean;
  include_text?: boolean;
  min_score?: number;
  min_likeness?: number;
  min_story?: number;
  min_continuity?: number;
  retry_count?: number;
}

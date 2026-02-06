/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import * as fs from 'fs';
import * as path from 'path';
import { FileSearchResult } from './types.js';

export class FileHandler {
  private static readonly OUTPUT_DIR = 'nanobanana-output';
  private static readonly SEARCH_PATHS = [
    process.cwd(),
    path.join(process.cwd(), 'images'),
    path.join(process.cwd(), 'input'),
    path.join(process.cwd(), this.OUTPUT_DIR),
    path.join(process.env.HOME || '~', 'Downloads'),
    path.join(process.env.HOME || '~', 'Desktop'),
  ];

  static ensureOutputDirectory(): string {
    const outputPath = path.join(process.cwd(), this.OUTPUT_DIR);
    this.ensureDirectory(outputPath);
    return outputPath;
  }

  static ensureDirectory(dirPath: string): void {
    if (!fs.existsSync(dirPath)) {
      fs.mkdirSync(dirPath, { recursive: true });
    }
  }

  static async saveTextFile(filePath: string, content: string): Promise<void> {
    await fs.promises.writeFile(filePath, content, 'utf-8');
  }

  static findInputFile(filename: string): FileSearchResult {
    if (path.isAbsolute(filename) && fs.existsSync(filename)) {
      return {
        found: true,
        filePath: filename,
        searchedPaths: [],
      };
    }

    const searchPaths = this.SEARCH_PATHS;

    for (const searchPath of searchPaths) {
      const fullPath = path.join(searchPath, filename);
      if (fs.existsSync(fullPath)) {
        return {
          found: true,
          filePath: fullPath,
          searchedPaths: searchPaths,
        };
      }
    }

    return {
      found: false,
      searchedPaths: searchPaths,
    };
  }

  static findInputDirectory(dirName: string): { found: boolean; dirPath?: string; files: string[] } {
    let targetPath = dirName;

    if (!path.isAbsolute(dirName)) {
      targetPath = path.join(process.cwd(), dirName);
    }

    if (fs.existsSync(targetPath) && fs.statSync(targetPath).isDirectory()) {
      try {
        const files = fs.readdirSync(targetPath)
          .filter(file => /\.(png|jpg|jpeg|webp)$/i.test(file))
          .map(file => path.join(targetPath, file));
        
        return {
          found: true,
          dirPath: targetPath,
          files: files
        };
      } catch (e) {
        console.error(`Error reading directory ${targetPath}:`, e);
      }
    }

    return {
      found: false,
      files: []
    };
  }

  static getSanitizedBaseName(prompt: string): string {
    // Create user-friendly filename from prompt
    let baseName = prompt
      .toLowerCase()
      .replace(/[^a-z0-9\s]/g, '') // Remove special characters
      .replace(/\s+/g, '_') // Replace spaces with underscores
      .substring(0, 64); // Limit to 64 characters

    if (!baseName) {
      baseName = 'generated_image';
    }
    return baseName;
  }

  static generateFilename(
    prompt: string,
    format: 'png' | 'jpeg' = 'png',
    index: number = 0,
  ): string {
    const baseName = this.getSanitizedBaseName(prompt);
    const extension = format === 'jpeg' ? 'jpg' : 'png';

    // Check for existing files and add counter if needed
    const outputPath = this.ensureOutputDirectory();
    let fileName = `${baseName}.${extension}`;
    let counter = index > 0 ? index : 1;

    while (fs.existsSync(path.join(outputPath, fileName))) {
      fileName = `${baseName}_${counter}.${extension}`;
      counter++;
    }

    return fileName;
  }

  static async saveImageFromBase64(
    base64Data: string,
    outputPath: string,
    filename: string,
  ): Promise<string> {
    const buffer = Buffer.from(base64Data, 'base64');
    const fullPath = path.join(outputPath, filename);

    await fs.promises.writeFile(fullPath, buffer);
    return fullPath;
  }

  static async readImageAsBase64(filePath: string): Promise<string> {
    const buffer = await fs.promises.readFile(filePath);
    return buffer.toString('base64');
  }

  static async readTextFile(filePath: string): Promise<string> {
    return await fs.promises.readFile(filePath, 'utf-8');
  }

  static findLatestFile(baseName: string): string | null {
    const outputPath = this.ensureOutputDirectory();
    
    try {
      const files = fs.readdirSync(outputPath);
      
      // Flexible regex: matches baseName followed by any combination of suffixes
      const escapedBaseName = baseName.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      const regex = new RegExp(`^${escapedBaseName}(_phase_1|_final|_\\d+)*\\.(png|jpg|jpeg)$`, 'i');

      const matches = files.filter((file: string) => regex.test(file));
      
      if (matches.length === 0) {
        return null;
      }

      // Sort by modification time, newest first
      const sortedMatches = matches.map((file: string) => {
        const fullPath = path.join(outputPath, file);
        return {
          path: fullPath,
          name: file,
          mtime: fs.statSync(fullPath).mtime.getTime()
        };
      }).sort((a: { mtime: number }, b: { mtime: number }) => b.mtime - a.mtime);

      console.error(`DEBUG - findLatestFile(${baseName}) found ${matches.length} matches. Newest: ${sortedMatches[0].name}`);
      return sortedMatches[0].path;
    } catch (error) {
      console.error('Error searching for latest file:', error);
      return null;
    }
  }

  static findPageFile(pageNumber: string): string | null {
      const outputPath = this.ensureOutputDirectory();
      try {
          const files = fs.readdirSync(outputPath);
          // Look for "manga_page_X_" pattern
          const regex = new RegExp(`^manga_page_${pageNumber}_.*(_final|_phase_1|_\\d+)*\\.(png|jpg|jpeg)$`, 'i');
          
          const matches = files.filter(f => regex.test(f));
          
          if (matches.length === 0) return null;
          
          // Prefer _final, then _phase_1, then others. Sort by priority then date.
          const sorted = matches.sort((a, b) => {
              // Priority sorting
              const aScore = a.includes('_final') ? 3 : (a.includes('_phase_1') ? 2 : 1);
              const bScore = b.includes('_final') ? 3 : (b.includes('_phase_1') ? 2 : 1);
              if (aScore !== bScore) return bScore - aScore;
              
              // Then date
              const aTime = fs.statSync(path.join(outputPath, a)).mtime.getTime();
              const bTime = fs.statSync(path.join(outputPath, b)).mtime.getTime();
              return bTime - aTime;
          });
          
          console.error(`DEBUG - findPageFile(${pageNumber}) found match: ${sorted[0]}`);
          return path.join(outputPath, sorted[0]);
      } catch (e) {
          return null;
      }
  }
}

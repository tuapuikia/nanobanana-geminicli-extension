import * as fs from 'fs';
import * as path from 'path';
import { FileHandler } from './fileHandler.js';

export class MemoryHandler {
  static async getMemoryFilePath(storyFile: string): Promise<string> {
    const storyDir = path.dirname(path.resolve(storyFile));
    return path.join(storyDir, 'manga_memory.md');
  }

  static async updateMemory(
    memoryPath: string,
    pageHeader: string,
    phase: number,
    status: 'PASSED' | 'FAILED',
    data?: { filePath?: string; reason?: string; failedPath?: string },
  ): Promise<void> {
    try {
      let content = '';
      if (fs.existsSync(memoryPath)) {
        content = await FileHandler.readTextFile(memoryPath);
      }

      // Escape header for regex
      const escapedHeader = pageHeader.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      const blockRegex = new RegExp(
        `(## ${escapedHeader}\\s*\\n)([\\s\\S]*?)(?=(\\n## |$))`,
        'i',
      );
      const match = content.match(blockRegex);

      let blockHeader = `## ${pageHeader}\n`;
      let blockContent = '';

      if (match) {
        blockHeader = match[1];
        blockContent = match[2];
      }

      // Logic:
      // If PASSED: Overwrite the specific Phase line.
      // If FAILED: Append a Failure log line.

      if (status === 'PASSED') {
        const phaseLineMarker = `- Phase ${phase}:`;
        const newLine = `${phaseLineMarker} \`${data?.filePath}\` [PASSED]`;
        const phaseRegex = new RegExp(
          `^\\s*${phaseLineMarker.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}.*$`,
          'm',
        );

        if (phaseRegex.test(blockContent)) {
          blockContent = blockContent.replace(phaseRegex, newLine);
        } else {
          blockContent = blockContent.trimEnd() + '\n' + newLine + '\n';
        }
      } else {
        // FAILED
        let failLine = `- Phase ${phase} Attempt: FAILED. Reason: ${data?.reason}`;
        if (data?.failedPath) {
            failLine += ` [FILE: \`${data.failedPath}\`]`;
        }
        
        // Prevent duplicate failure logs if exactly the same
        if (!blockContent.includes(failLine)) {
             blockContent = blockContent.trimEnd() + '\n' + failLine + '\n';
        }
      }

      let newContent = '';
      if (match) {
        newContent = content.replace(blockRegex, blockHeader + blockContent);
      } else {
        newContent = content + '\n\n' + blockHeader + blockContent;
      }

      await FileHandler.saveTextFile(memoryPath, newContent.trim());
      console.error(
        `DEBUG - Updated Memory: ${pageHeader} Phase ${phase} [${status}]`,
      );
    } catch (e) {
      console.error(`DEBUG - Failed to update memory:`, e);
    }
  }

  static async getFailures(
    memoryPath: string,
    pageHeader: string,
  ): Promise<{ reasons: string[]; failedPaths: string[] }> {
    try {
      if (!fs.existsSync(memoryPath)) return { reasons: [], failedPaths: [] };
      const content = await FileHandler.readTextFile(memoryPath);

      const escapedHeader = pageHeader.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      const blockRegex = new RegExp(
        `## ${escapedHeader}\\s*\\n([\\s\\S]*?)(?=(\\n## |$))`,
        'i',
      );
      const match = content.match(blockRegex);

      if (!match) return { reasons: [], failedPaths: [] };

      const blockContent = match[1];
      const reasons: string[] = [];
      const failedPaths: string[] = [];
      
      const lines = blockContent.split('\n');
      for (const line of lines) {
          const failMatch = line.match(/- Phase \d+ Attempt: FAILED\. Reason: (.*?)(?: \[FILE: `([^`]+)`\])?$/);
          if (failMatch) {
              reasons.push(failMatch[1].trim());
              if (failMatch[2] && fs.existsSync(failMatch[2])) {
                  failedPaths.push(failMatch[2]);
              }
          }
      }
      return { 
          reasons: [...new Set(reasons)],
          failedPaths: [...new Set(failedPaths)]
      };
    } catch (e) {
      return { reasons: [], failedPaths: [] };
    }
  }

  static async checkMemory(
    memoryPath: string,
    pageHeader: string,
  ): Promise<{ phase1?: string; phase2?: string }> {
    try {
      if (!fs.existsSync(memoryPath)) return {};
      const content = await FileHandler.readTextFile(memoryPath);

      const escapedHeader = pageHeader.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      const blockRegex = new RegExp(
        `## ${escapedHeader}\\s*\\n([\\s\\S]*?)(?=(\\n## |$))`,
        'i',
      );
      const match = content.match(blockRegex);

      if (!match) return {};

      const blockContent = match[1];
      const result: { phase1?: string; phase2?: string } = {};

      // Extract Phase 1
      const p1Match = blockContent.match(/- Phase 1: `([^`]+)` \[PASSED\]/);
      if (p1Match && fs.existsSync(p1Match[1])) {
        result.phase1 = p1Match[1];
      }

      // Extract Phase 2
      const p2Match = blockContent.match(/- Phase 2: `([^`]+)` \[PASSED\]/);
      if (p2Match && fs.existsSync(p2Match[1])) {
        result.phase2 = p2Match[1];
      }

      return result;
    } catch (e) {
      console.error(`DEBUG - Failed to read memory:`, e);
      return {};
    }
  }
}
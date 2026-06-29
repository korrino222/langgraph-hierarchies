import { readdirSync, rmSync } from 'node:fs';
import { join } from 'node:path';

const distDir = new URL('../dist', import.meta.url).pathname;

function pruneMarkdownFiles(dir) {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const path = join(dir, entry.name);
    if (entry.isDirectory()) {
      pruneMarkdownFiles(path);
      continue;
    }
    if (entry.name.endsWith('.md') || entry.name.endsWith('.mdx')) {
      rmSync(path);
    }
  }
}

pruneMarkdownFiles(distDir);

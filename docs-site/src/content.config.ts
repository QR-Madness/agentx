import { defineCollection, z } from 'astro:content';
import { glob } from 'astro/loaders';

// Docs are plain Markdown migrated as-is (no frontmatter); titles come from the first H1
// or the nav label, so all schema fields are optional.
const docs = defineCollection({
  loader: glob({ pattern: '**/*.md', base: './src/content/docs' }),
  schema: z.object({
    title: z.string().optional(),
    description: z.string().optional(),
  }),
});

export const collections = { docs };

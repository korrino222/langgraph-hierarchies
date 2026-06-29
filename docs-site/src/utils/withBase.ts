/** Prefix an internal docs path with the Astro `base` URL (GitHub Pages subpath). */
export function withBase(path: string): string {
  if (!path || path.startsWith('http://') || path.startsWith('https://')) {
    return path;
  }
  if (path.startsWith('#')) {
    return path;
  }

  const base = import.meta.env.BASE_URL;
  if (path === '/') {
    return base;
  }

  const normalized = path.startsWith('/') ? path.slice(1) : path;
  return `${base}${normalized}`;
}

/**
 * sigma reads `WebGLRenderingContext` and `WebGL2RenderingContext` constants while its module is evaluated, and happy-dom
 * has no WebGL. A test that imports sigma (its Camera, its edge programs) imports this module
 * FIRST, so the constants exist; nothing is ever drawn. An empty object is enough: only the
 * property lookups run, and they yield undefined.
 */
const globals = globalThis as Record<string, unknown>
for (const name of ['WebGLRenderingContext', 'WebGL2RenderingContext']) {
  if (!(name in globals)) globals[name] = {}
}

export {}

import { useEffect, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js'
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js'
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'

import type { Band, BandState, Hsl } from '@/lib/lens'
import {
  DEFAULT_BAND_COUNT,
  NEUTRAL,
  bandLayers,
  bandOfLayer,
  blendBand,
  classColor,
  crossoverLayer,
  cssColor,
  hasLensData,
} from '@/lib/lens'
import type { JobProgress, Trace } from '@/lib/api-types'
import type { Atlas } from '@/lib/atlas'
import {
  atlasLayers,
  loadAtlas,
  namedAreaCount,
  positionBuffer,
  sourceClaim,
} from '@/lib/atlas'
import type { LitNode, LitScope, LitSet } from '@/lib/lit'
import {
  AGGREGATION,
  BOS_REASON,
  atlasAgreement,
  coverageNote,
  litSet,
  prominence,
} from '@/lib/lit'
import { KdTree, rayPoints } from '@/lib/kdtree'
import type { RunState } from '@/hooks/useTrace'

// Deterministic lattice hash + trilinear interpolation ("value noise").
// Needs to be spatially coherent (unlike a plain per-point hash) so that
// neighboring vertices move together into smooth, rounded gyri instead of spikes.
function hash3(x: number, y: number, z: number): number {
  const s = Math.sin(x * 127.1 + y * 311.7 + z * 74.7) * 43758.5453123
  return s - Math.floor(s)
}

function fade(t: number): number {
  return t * t * t * (t * (t * 6 - 15) + 10)
}

function valueNoise3(x: number, y: number, z: number): number {
  const xi = Math.floor(x)
  const yi = Math.floor(y)
  const zi = Math.floor(z)
  const xf = x - xi
  const yf = y - yi
  const zf = z - zi
  const u = fade(xf)
  const v = fade(yf)
  const w = fade(zf)

  const c000 = hash3(xi, yi, zi)
  const c100 = hash3(xi + 1, yi, zi)
  const c010 = hash3(xi, yi + 1, zi)
  const c110 = hash3(xi + 1, yi + 1, zi)
  const c001 = hash3(xi, yi, zi + 1)
  const c101 = hash3(xi + 1, yi, zi + 1)
  const c011 = hash3(xi, yi + 1, zi + 1)
  const c111 = hash3(xi + 1, yi + 1, zi + 1)

  const x00 = THREE.MathUtils.lerp(c000, c100, u)
  const x10 = THREE.MathUtils.lerp(c010, c110, u)
  const x01 = THREE.MathUtils.lerp(c001, c101, u)
  const x11 = THREE.MathUtils.lerp(c011, c111, u)
  const y0 = THREE.MathUtils.lerp(x00, x10, v)
  const y1 = THREE.MathUtils.lerp(x01, x11, v)

  return THREE.MathUtils.lerp(y0, y1, w) * 2 - 1
}

// Same ellipsoid shaping used inside buildBrainGeometry, factored out so
// anatomical boundary lines (drawn separately) sit flush with the surface.
function shapeEllipsoid(v: THREE.Vector3): THREE.Vector3 {
  v.x *= 0.86
  v.y *= 0.66
  v.z *= 1.18

  // Frontal lobe: the anterior pole is noticeably narrower and rounder than
  // the back of the brain, so pinch width and height in as z increases.
  if (v.z > 0.15) {
    const t = THREE.MathUtils.clamp((v.z - 0.15) / 0.85, 0, 1)
    const taper = 1 - t * t * 0.36
    v.x *= taper
    v.y *= 1 - t * t * 0.14
  }

  // Occipital lobe: fuller/rounder through the back before pinching at the pole.
  if (v.z < -0.1) {
    const t = THREE.MathUtils.clamp((-v.z - 0.1) / 0.9, 0, 1)
    v.x *= 1 + Math.sin(t * Math.PI) * 0.09 * (1 - t)
    const pinch = 1 - Math.pow(t, 4) * 0.22
    v.x *= pinch
    v.y *= pinch
  }

  // Temporal lobes: bulge outward and droop downward along the lower sides,
  // giving the characteristic flap that hangs below the lateral (Sylvian) fissure
  // instead of a plain ellipsoid waist.
  const temporalBand = Math.exp(-Math.pow((v.y + 0.16) / 0.26, 2)) * Math.exp(-Math.pow(v.z / 0.65, 2))
  v.x *= 1 + temporalBand * 0.24
  v.y -= temporalBand * 0.15

  // Twin-hemisphere crown: each hemisphere domes up and out away from the
  // midline near the top, so the brain reads as two lobes from above rather
  // than one smooth dome (the longitudinal fissure carved in later deepens this).
  const crownBand = Math.exp(-Math.pow((v.y - 0.32) / 0.34, 2))
  const hemisphereLift = Math.exp(-Math.pow((Math.abs(v.x) - 0.3) / 0.22, 2))
  v.y += crownBand * hemisphereLift * 0.06

  if (v.y < -0.3) {
    v.y = -0.3 + (v.y + 0.3) * 0.45
  }

  return v
}

// A point on the brain's outer shell given azimuth (0 = front, 90 = right side,
// 180 = back) and elevation (0 = equator, 90 = crown) in degrees, lifted
// slightly outward so lines drawn with it sit just above the glass surface.
function surfacePoint(azimuthDeg: number, elevationDeg: number, lift = 1.02): THREE.Vector3 {
  const az = THREE.MathUtils.degToRad(azimuthDeg)
  const el = THREE.MathUtils.degToRad(elevationDeg)
  const horizontal = Math.cos(el)
  const v = new THREE.Vector3(horizontal * Math.sin(az), Math.sin(el), horizontal * Math.cos(az)).multiplyScalar(1.4)
  return shapeEllipsoid(v).multiplyScalar(lift)
}

function lobeBoundaryCurve(points: THREE.Vector3[]): THREE.Line {
  const curve = new THREE.CatmullRomCurve3(points)
  const geometry = new THREE.BufferGeometry().setFromPoints(curve.getPoints(40))
  const material = new THREE.LineBasicMaterial({ color: 0xe6e8eb, transparent: true, opacity: 0.55 })
  return new THREE.Line(geometry, material)
}

function buildLobeBoundaries(): THREE.Group {
  const group = new THREE.Group()

  // Longitudinal fissure, traced over the crown from front to back.
  const fissurePoints: THREE.Vector3[] = []
  for (let el = -5; el <= 84; el += 8) fissurePoints.push(surfacePoint(0, el))
  fissurePoints.push(surfacePoint(0, 90))
  for (let el = 84; el >= -5; el -= 8) fissurePoints.push(surfacePoint(180, el))
  group.add(lobeBoundaryCurve(fissurePoints))

  // Central sulcus (frontal / parietal boundary) — one per hemisphere.
  for (const side of [1, -1]) {
    const points = [
      surfacePoint(side * 95, 82),
      surfacePoint(side * 88, 55),
      surfacePoint(side * 80, 25),
      surfacePoint(side * 72, -5),
      surfacePoint(side * 66, -20),
    ]
    group.add(lobeBoundaryCurve(points))
  }

  // Lateral (Sylvian) sulcus, separating the temporal lobe below.
  for (const side of [1, -1]) {
    const points = [
      surfacePoint(side * 35, -5),
      surfacePoint(side * 60, -18),
      surfacePoint(side * 90, -22),
      surfacePoint(side * 120, -16),
      surfacePoint(side * 140, -8),
    ]
    group.add(lobeBoundaryCurve(points))
  }

  // Parieto-occipital sulcus, near the back of the crown.
  for (const side of [1, -1]) {
    const points = [surfacePoint(side * 155, 68), surfacePoint(side * 172, 48), surfacePoint(side * 186, 30)]
    group.add(lobeBoundaryCurve(points))
  }

  return group
}

function buildBrainGeometry(): THREE.BufferGeometry {
  const geometry = new THREE.IcosahedronGeometry(1.4, 20)
  const position = geometry.attributes.position
  const vertex = new THREE.Vector3()

  for (let i = 0; i < position.count; i++) {
    vertex.fromBufferAttribute(position, i)
    const normalized = vertex.clone().normalize()

    shapeEllipsoid(vertex)

    // Cerebellum: a distinct, tightly-rounded bulge tucked under the occipital
    // lobe, textured with finer folia than the broad cerebral gyri.
    const cerebellum = new THREE.Vector3(0, -0.46, -0.92)
    const cerebellumDist = vertex.distanceTo(cerebellum)
    const cerebellumWeight = Math.max(0, 1 - cerebellumDist / 0.36) ** 2
    const cerebellumFolia = valueNoise3(
      normalized.x * 16 + 5.2,
      normalized.y * 16 + 8.1,
      normalized.z * 16 + 2.4,
    )
    const cerebellumBump = cerebellumWeight * (0.2 + cerebellumFolia * 0.025)

    // Transverse fissure: a groove just above the cerebellum separating it
    // from the occipital lobe, so the two read as separate structures.
    const grooveCenter = new THREE.Vector3(0, -0.16, -0.98)
    const grooveDist = vertex.distanceTo(grooveCenter)
    const transverseGroove = Math.max(0, 1 - grooveDist / 0.24) ** 2 * 0.055

    // Layered coherent noise for gyri (broad rounded folds + finer wrinkles).
    const broad = valueNoise3(
      normalized.x * 2.6 + 4.1,
      normalized.y * 2.6 + 1.7,
      normalized.z * 2.6 + 9.3,
    )
    const fine = valueNoise3(
      normalized.x * 6 - 3.5,
      normalized.y * 6 + 6.6,
      normalized.z * 6 - 1.2,
    )
    const fold = broad * 0.1 + fine * 0.042

    // Longitudinal fissure separating the two hemispheres — deep on top, fading toward
    // the sides and disappearing on the underside.
    const topFactor = THREE.MathUtils.clamp((vertex.y + 0.05) / 0.55, 0, 1)
    // Fade the groove out near the front/back tips, where x is ~0 for every
    // vertex regardless of the fissure — without this the whole tip pinches in.
    const radialFromAxis = Math.hypot(vertex.x, vertex.z)
    const tipFade = THREE.MathUtils.clamp(radialFromAxis / 0.35, 0, 1)
    const fissure = Math.exp(-Math.pow(vertex.x * 7.5, 2)) * 0.13 * topFactor * tipFade

    const displaced = vertex
      .clone()
      .addScaledVector(normalized, fold - fissure + cerebellumBump - transverseGroove)

    position.setXYZ(i, displaced.x, displaced.y, displaced.z)
  }

  geometry.computeVertexNormals()
  return geometry
}

// --------------------------------------------------------------------------
// layer bands
// --------------------------------------------------------------------------

// How thick a slab of vertices counts as sitting on a band boundary, in the
// geometry's own units. Wide enough that the ring is continuous on a wrinkled
// surface, narrow enough that it still reads as a line.
const RING_SLAB = 0.045
const RING_SEGMENTS = 72

// Depth runs front to back: band 0 (the model's first layers) at the frontal
// pole, the last band at the occipital pole. Front-to-back rather than any
// other axis because the grid already reads "input -> output" left to right,
// so the brain inherits the same directional convention.
function depthRange(position: THREE.BufferAttribute): { front: number; back: number } {
  let front = -Infinity
  let back = Infinity
  for (let i = 0; i < position.count; i++) {
    const z = position.getZ(i)
    if (z > front) front = z
    if (z < back) back = z
  }
  return { front, back }
}

/**
 * Which visual band a depth falls in. `front` maps to band 0.
 *
 * This is a plain geometric slicing of the shell and carries no anatomical
 * meaning — see the legend the component renders, and design.md on why the
 * sulcus lines are pushed into the background whenever bands are shown.
 */
function bandForDepth(z: number, front: number, back: number, bandCount: number): number {
  const span = front - back
  if (span <= 0) return 0
  const t = (front - z) / span
  return Math.min(bandCount - 1, Math.max(0, Math.floor(t * bandCount)))
}

/** Per-vertex band index, computed once per band count rather than per frame. */
function assignVertexBands(
  position: THREE.BufferAttribute,
  front: number,
  back: number,
  bandCount: number,
): Uint8Array {
  const bands = new Uint8Array(position.count)
  for (let i = 0; i < position.count; i++) {
    bands[i] = bandForDepth(position.getZ(i), front, back, bandCount)
  }
  return bands
}

/**
 * A closed loop hugging the surface at depth `z`, for drawing a band boundary.
 *
 * Sampled from the geometry's own vertices rather than drawn as an analytic
 * ellipse: the shell has gyri, a cerebellum bulge and a longitudinal fissure
 * carved into it, so a ring computed from the ellipsoid formula would float
 * off the surface in some places and sink into it in others. Returns null near
 * the poles, where a constant-z slice has too little of the surface in it to
 * describe a ring.
 */
function ringPointsAtDepth(
  position: THREE.BufferAttribute,
  z: number,
): THREE.Vector3[] | null {
  const radii = new Float32Array(RING_SEGMENTS).fill(-1)
  const vertex = new THREE.Vector3()

  for (let i = 0; i < position.count; i++) {
    vertex.fromBufferAttribute(position, i)
    if (Math.abs(vertex.z - z) > RING_SLAB) continue
    const angle = Math.atan2(vertex.y, vertex.x)
    const bin = Math.min(
      RING_SEGMENTS - 1,
      Math.floor(((angle + Math.PI) / (Math.PI * 2)) * RING_SEGMENTS),
    )
    const radius = Math.hypot(vertex.x, vertex.y)
    if (radius > radii[bin]) radii[bin] = radius
  }

  const filled = radii.reduce((n, r) => (r > 0 ? n + 1 : n), 0)
  if (filled < RING_SEGMENTS * 0.6) return null // too close to a pole to ring

  // Bridge the gaps left by empty angular bins so the loop stays closed.
  for (let bin = 0; bin < RING_SEGMENTS; bin++) {
    if (radii[bin] > 0) continue
    let before = bin
    let after = bin
    while (radii[(before + RING_SEGMENTS) % RING_SEGMENTS] <= 0) before -= 1
    while (radii[after % RING_SEGMENTS] <= 0) after += 1
    const a = radii[((before % RING_SEGMENTS) + RING_SEGMENTS) % RING_SEGMENTS]
    const b = radii[after % RING_SEGMENTS]
    radii[bin] = (a + b) / 2
  }

  const points: THREE.Vector3[] = []
  for (let bin = 0; bin <= RING_SEGMENTS; bin++) {
    const index = bin % RING_SEGMENTS
    const angle = (index / RING_SEGMENTS) * Math.PI * 2 - Math.PI
    const radius = radii[index] * 1.012 // lift clear of the shell
    points.push(new THREE.Vector3(Math.cos(angle) * radius, Math.sin(angle) * radius, z))
  }
  return points
}

// --------------------------------------------------------------------------
// activity: what the brain shows while a job is in flight
// --------------------------------------------------------------------------

/**
 * What the brain is allowed to claim is happening right now.
 *
 * `unknown` is the honest state for "a job exists and is working, but nothing
 * has told us where it is" — including the warm-up window, when the service is
 * still answering 503 and no job exists at all. `generating` advances a token
 * counter and deliberately does *not* light a layer: the capture loop computes
 * a whole forward pass per token, so no layer is ever singly "executing" (see
 * design.md). `lens` is the one phase that really does walk depth, so it is
 * the only one that drives the band sweep.
 */
type ActivityKind = 'idle' | 'unknown' | 'generating' | 'lens'

interface Activity {
  kind: ActivityKind
  /** Band index the sweep is allowed to reach — never exceeded. */
  targetBand: number
  done: number
  total: number
}

const IDLE: Activity = { kind: 'idle', targetBand: 0, done: 0, total: 0 }

/** What one band is showing for the selected position. */
interface BandView {
  band: Band
  /** null when there is no trace, or no lens readout anywhere in the band. */
  state: BandState | null
  color: Hsl
  isCrossover: boolean
  containsSelection: boolean
}

export interface BrainProps {
  trace: Trace | null
  /** The (layer, token) the grid and the brain share. */
  selection: { layer: number; position: number } | null
  status: RunState
  progress: JobProgress | null
  /** Clicking a band selects its first layer, keeping both surfaces in step. */
  onSelectLayer?: (layer: number) => void
}

// --------------------------------------------------------------------------
// the node cloud
// --------------------------------------------------------------------------

/**
 * How a node is drawn when nothing has lit it.
 *
 * Dim enough to read as texture rather than as data — an inactive node says
 * "a feature exists here", and the brain is full of them before any prompt is
 * sent. Activation lighting is layered on top of this, so this is the floor
 * and never the whole story.
 */
const NODE_IDLE_COLOR = new THREE.Color(0x6e7681)
const NODE_IDLE_OPACITY = 0.5
const NODE_SIZE = 0.016

/**
 * The point cloud, as one `THREE.Points` in one draw call.
 *
 * `sizeAttenuation` so nodes shrink with depth and the cloud reads as a
 * volume rather than a flat spray. Additive blending against the black
 * background, and `depthWrite: false` so the translucent shell in front of a
 * node does not punch a hole in it.
 */
function buildNodeCloud(atlas: Atlas): THREE.Points {
  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(positionBuffer(atlas.nodes), 3))

  const material = new THREE.PointsMaterial({
    color: NODE_IDLE_COLOR,
    size: NODE_SIZE,
    sizeAttenuation: true,
    transparent: true,
    opacity: NODE_IDLE_OPACITY,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  })

  const points = new THREE.Points(geometry, material)
  // The cloud fills the shell's own volume, so it must never be culled by a
  // bounding sphere computed before the shell rotates into place.
  points.frustumCulled = false
  return points
}

// --------------------------------------------------------------------------
// the active cloud: the features this trace lit
// --------------------------------------------------------------------------

/**
 * Nodes lit by a trace, drawn as a second `THREE.Points` over the idle cloud.
 *
 * A separate object rather than a recolouring of the idle one, because they
 * are not the same set: the idle asset is a 20,000-node *sample* of the atlas,
 * and the features a trace reports are mostly not in it. Positions here come
 * from `trace.layout` — the atlas's answer for these exact features — so what
 * lights is what fired, not whichever nodes happened to be sampled.
 *
 * `activation` rides as a per-vertex attribute and the shader maps it to both
 * size and brightness. That is what keeps 4.3's second clause true: a dim node
 * beside a bright cluster stays dim, because every node's appearance is a pure
 * function of its own attribute and nothing is blurred or pooled across
 * neighbours.
 */
// Sized well clear of the idle cloud's 0.016. At the same size a lit node is
// only a colour difference among 20,000 points and does not read as data.
const ACTIVE_COLOR = new THREE.Color(0x7ee787)
const ACTIVE_MIN_SIZE = 0.034
const ACTIVE_MAX_SIZE = 0.11

/**
 * How far the idle cloud drops back once something is lit.
 *
 * The idle sample is context — "features exist here" — and the lit set is the
 * finding. At cell scope the lit set is ~16 nodes against 20,000, so without
 * this the data is a rounding error on the texture behind it.
 */
const NODE_IDLE_OPACITY_LIT = 0.13

// -- picking ---------------------------------------------------------------

/** Slightly outside the shell, so a node right at the surface is reachable. */
const SHELL_PICK_RADIUS = 1.6
const PICK_STEPS = 96
/** How near the ray a node must pass to count as hovered, in shell units. */
const PICK_RADIUS = 0.055

/**
 * Where a ray enters and leaves a sphere at the origin, or null if it misses.
 *
 * Used to spend every pick sample inside the volume that holds nodes instead
 * of on the empty units between the camera and the brain.
 */
function intersectSphere(
  origin: THREE.Vector3,
  direction: THREE.Vector3,
  radius: number,
): [number, number] | null {
  const b = origin.dot(direction)
  const c = origin.dot(origin) - radius * radius
  const discriminant = b * b - c
  if (discriminant < 0) return null
  const root = Math.sqrt(discriminant)
  const near = -b - root
  const far = -b + root
  if (far < 0) return null
  return [Math.max(near, 0), far]
}

const ACTIVE_VERTEX_SHADER = `
attribute float prominence;
varying float vProminence;
void main() {
  vProminence = prominence;
  vec4 viewPosition = modelViewMatrix * vec4(position, 1.0);
  // sizeAttenuation, by hand: nodes shrink with distance so the cloud reads
  // as a volume rather than a flat spray.
  gl_PointSize = mix(${ACTIVE_MIN_SIZE.toFixed(3)}, ${ACTIVE_MAX_SIZE.toFixed(3)}, prominence)
    * (300.0 / -viewPosition.z);
  gl_Position = projectionMatrix * viewPosition;
}
`

const ACTIVE_FRAGMENT_SHADER = `
uniform vec3 color;
varying float vProminence;
void main() {
  // Round sprite, soft edge. Discarding outside the disc keeps nodes from
  // reading as squares at close range.
  vec2 offset = gl_PointCoord - vec2(0.5);
  float radius = length(offset);
  if (radius > 0.5) discard;
  float edge = smoothstep(0.5, 0.15, radius);
  gl_FragColor = vec4(color, edge * mix(0.35, 1.0, vProminence));
}
`

function buildActiveCloud(nodes: LitNode[], maxActivation: number): THREE.Points {
  const positions = new Float32Array(nodes.length * 3)
  const prominences = new Float32Array(nodes.length)
  for (let i = 0; i < nodes.length; i++) {
    positions[i * 3] = nodes[i].x
    positions[i * 3 + 1] = nodes[i].y
    positions[i * 3 + 2] = nodes[i].z
    prominences[i] = prominence(nodes[i].activation, maxActivation)
  }

  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
  geometry.setAttribute('prominence', new THREE.BufferAttribute(prominences, 1))

  const material = new THREE.ShaderMaterial({
    uniforms: { color: { value: ACTIVE_COLOR } },
    vertexShader: ACTIVE_VERTEX_SHADER,
    fragmentShader: ACTIVE_FRAGMENT_SHADER,
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  })

  const points = new THREE.Points(geometry, material)
  points.frustumCulled = false
  return points
}

/** Mutable three.js handles, built once and reused across prop changes. */
interface SceneHandles {
  renderer: THREE.WebGLRenderer
  composer: EffectComposer
  camera: THREE.PerspectiveCamera
  brainGroup: THREE.Group
  brainMesh: THREE.Mesh
  geometry: THREE.BufferGeometry
  brainstemGeometry: THREE.BufferGeometry
  material: THREE.MeshPhysicalMaterial
  lobeBoundaries: THREE.Group
  bandRings: THREE.Group
  /** The atlas's node cloud, once an atlas has loaded. Null until then. */
  nodeCloud: THREE.Points | null
  /** The features the current trace lit, at the current scope. */
  activeCloud: THREE.Points | null
  front: number
  back: number
  vertexBands: Uint8Array
  vertexBandCount: number
  /** Band colours as painted, before any activity glow is layered on. */
  baseColors: Float32Array
  /** Eased sweep position, in fractional band units. Never exceeds the target. */
  sweep: number
  /** Whether the last frame wrote a glow that has to be cleared. */
  glowing: boolean
}

export function Brain({ trace, selection, status, progress, onSelectLayer }: BrainProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const handlesRef = useRef<SceneHandles | null>(null)
  const [hoveredBand, setHoveredBand] = useState<number | null>(null)

  // The atlas is a fixed table fetched once, not derived from the trace.
  // `undefined` while the fetch is in flight, `null` when there is no atlas to
  // be had — the two are different states and the legend distinguishes them,
  // because "loading" and "this deployment has no atlas" are different answers.
  const [atlas, setAtlas] = useState<Atlas | null | undefined>(undefined)

  useEffect(() => {
    let cancelled = false
    loadAtlas().then((loaded) => {
      if (!cancelled) setAtlas(loaded)
    })
    return () => {
      cancelled = true
    }
  }, [])

  // How much of the trace lights at once.
  //
  // Trace scope is the default for a legibility reason worth writing down: a
  // cell lights ~16 nodes against an idle cloud of 20,000, which is 0.08% of
  // the points on screen and reads as nothing at all. The whole trace lights
  // ~1,350, which reads as structure you can then narrow. Starting at the
  // narrowest scope was starting on an apparently empty brain.
  //
  // It aggregates, and the panel says so — `aggregation` is stated rather
  // than left to be inferred, which is what makes the wider default honest.
  const [scope, setScope] = useState<LitScope>('trace')

  // Which node the pointer is over, if any.
  const [hoveredNode, setHoveredNode] = useState<LitNode | null>(null)

  // The features this trace lit. Positions come from `trace.layout`, never
  // from the atlas asset — see litSet's own note on why that distinction is
  // not cosmetic.
  const lit = useMemo<LitSet>(() => litSet(trace, scope, selection), [trace, scope, selection])

  // Whether the trace's positions correspond to the atlas on screen. A
  // mismatch is not a warning to bury: the areas, names and neighbourhoods on
  // screen belong to a different map, so the positions may not be presented
  // as this atlas's.
  const agreement = useMemo(() => atlasAgreement(trace, atlas), [trace, atlas])
  const positionsUsable = agreement.kind === 'match' || agreement.kind === 'no-view-atlas'

  // Read by the animation loop every frame. A ref rather than state so a new
  // progress reading never restarts the loop or rebuilds the scene.
  const activityRef = useRef<Activity>(IDLE)

  const lensAvailable = trace !== null && hasLensData(trace)

  // What each band is showing. Recomputed when the trace or the selected token
  // changes; never triggers a geometry or renderer rebuild.
  const bandViews = useMemo<BandView[]>(() => {
    if (trace === null || trace.steps.length === 0) {
      return bandLayers(DEFAULT_BAND_COUNT, DEFAULT_BAND_COUNT).map((band) => ({
        band,
        state: null,
        color: NEUTRAL,
        isCrossover: false,
        containsSelection: false,
      }))
    }

    const bands = bandLayers(trace.n_layers, DEFAULT_BAND_COUNT)
    const crossover = crossoverLayer(trace)
    const crossoverBand = crossover === null ? -1 : bandOfLayer(bands, crossover)
    const position = Math.min(selection?.position ?? trace.steps.length - 1, trace.steps.length - 1)
    const step = trace.steps[position]
    const selectedBand = selection === null ? -1 : bandOfLayer(bands, selection.layer)

    return bands.map((band) => {
      const state = lensAvailable && step ? blendBand(step, band) : null
      return {
        band,
        state,
        // A band with no lens readout stays neutral rather than borrowing a
        // class colour it has no evidence for.
        color: state === null || state.klass === null ? NEUTRAL : classColor(state.klass, state.confidence),
        isCrossover: band.index === crossoverBand,
        containsSelection: band.index === selectedBand,
      }
    })
  }, [trace, selection, lensAvailable])

  const bandCount = bandViews.length

  // The activity the brain is entitled to show, derived from the job's own
  // reported state. Nothing here interpolates *past* a reading; the loop eases
  // toward it and stops there.
  const activity = useMemo<Activity>(() => {
    if (status === 'error' || status === 'idle' || status === 'done') return IDLE
    // Submitted, or warming up, and nothing reported yet.
    if (status === 'warming' || status === 'pending' || progress === null) {
      return { kind: 'unknown', targetBand: 0, done: 0, total: 0 }
    }
    if (progress.phase === 'generating') {
      return {
        kind: 'generating',
        targetBand: 0,
        done: progress.done,
        total: progress.total,
      }
    }
    // The lens phase counts layers, so a reading maps onto a band.
    const layer = Math.max(0, progress.done - 1)
    const perBand = progress.total / Math.max(bandCount, 1)
    return {
      kind: 'lens',
      targetBand: Math.min(bandCount - 1, Math.floor(layer / Math.max(perBand, 1))),
      done: progress.done,
      total: progress.total,
    }
  }, [status, progress, bandCount])

  useEffect(() => {
    activityRef.current = activity
    const handles = handlesRef.current
    if (!handles) return
    // A fresh run restarts the sweep from the front of the brain.
    if (activity.kind === 'idle' || activity.kind === 'unknown') handles.sweep = 0
  }, [activity])

  // -- setup: runs once, and owns everything expensive ---------------------
  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100)
    camera.position.set(0, 2, 4.6)
    camera.lookAt(0, -0.15, 0)

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false })
    // The surface tone the canvas sits inside, not #000000: pure black
    // vibrates against the near-blacks around it and reads as a hole rather
    // than the deepest layer of the same system.
    renderer.setClearColor(0x101216, 1)
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    container.appendChild(renderer.domElement)

    // Drag to orbit, wheel to zoom. Without this the cloud is a fixed
    // projection of a 3-d layout, which is the one thing a 3-d layout must not
    // be: a node hidden behind another is unreachable, and depth reads as
    // overlap. Panning stays off — the atlas has no meaningful origin to move
    // away from, and an off-centre brain is just lost.
    const controls = new OrbitControls(camera, renderer.domElement)
    controls.target.set(0, -0.15, 0)
    controls.enablePan = false
    controls.enableDamping = true
    controls.dampingFactor = 0.08
    controls.rotateSpeed = 0.6
    controls.zoomSpeed = 0.8
    // Close enough to resolve one node, far enough to see the whole shell.
    controls.minDistance = 1.9
    controls.maxDistance = 9
    controls.update()

    // The idle drift is an attract loop, not a feature. The moment the pointer
    // takes hold it stops for good: a target that keeps moving under the
    // cursor cannot be hovered, which is what made node inspection unusable.
    let userDriven = false
    controls.addEventListener('start', () => {
      userDriven = true
    })

    const brainGroup = new THREE.Group()
    scene.add(brainGroup)

    const geometry = buildBrainGeometry()
    const position = geometry.attributes.position as THREE.BufferAttribute

    // Band colour lives on the geometry as a vertex attribute, so repainting
    // bands is a buffer write rather than a rebuild of the (expensive,
    // noise-displaced) shell. Seeded to the glass tone the brain has with no
    // trace loaded.
    const colors = new Float32Array(position.count * 3)
    colors.fill(0.902) // #E6E8EB, the interface's primary tone — never pure white.
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3))

    // Translucent glass shell — see-through so the underlying fold structure
    // (and the layer bands painted onto it) reads clearly against black.
    const material = new THREE.MeshPhysicalMaterial({
      color: new THREE.Color(0xffffff), // white: the vertex colours do the tinting
      vertexColors: true,
      transparent: true,
      opacity: 0.38,
      roughness: 0.45,
      metalness: 0,
      clearcoat: 0.25,
      clearcoatRoughness: 0.5,
      side: THREE.DoubleSide,
      depthWrite: false,
    })
    const brainMesh = new THREE.Mesh(geometry, material)
    brainGroup.add(brainMesh)

    // Lobe boundary lines (central sulcus, Sylvian fissure, longitudinal
    // fissure, ...). They are silhouette, not data: the component drops their
    // opacity whenever bands are showing, so nobody reads a named anatomical
    // region as the site of a layer range.
    const lobeBoundaries = buildLobeBoundaries()
    brainGroup.add(lobeBoundaries)

    // Band boundary rings, rebuilt when the band layout or the crossover
    // layer changes.
    const bandRings = new THREE.Group()
    brainGroup.add(bandRings)

    // Brainstem: a short tapered stalk beneath the cerebellum, so the
    // silhouette reads as a brain rather than a bare cerebral mass. Uses its
    // own untinted material so band colours do not bleed onto it.
    const brainstemGeometry = new THREE.CylinderGeometry(0.075, 0.12, 0.4, 16)
    const brainstemMaterial = material.clone()
    brainstemMaterial.vertexColors = false
    brainstemMaterial.color = new THREE.Color(0xe6e8eb)
    const brainstemMesh = new THREE.Mesh(brainstemGeometry, brainstemMaterial)
    brainstemMesh.position.set(0, -0.62, -0.45)
    brainstemMesh.rotation.x = THREE.MathUtils.degToRad(18)
    brainGroup.add(brainstemMesh)

    const ambient = new THREE.AmbientLight(0xffffff, 0.55)
    const keyLight = new THREE.DirectionalLight(0xffffff, 1.5)
    keyLight.position.set(3, 3.5, 3)
    const fillLight = new THREE.DirectionalLight(0xaaaaaa, 0.6)
    fillLight.position.set(-3, 1, 2)
    const rimLight = new THREE.DirectionalLight(0xffffff, 1.2)
    rimLight.position.set(-2, -1, -4)
    scene.add(ambient, keyLight, fillLight, rimLight)

    // RenderPass and OutputPass only: OutputPass is what applies the tone
    // mapping and sRGB conversion, so it earns its place. There is no bloom —
    // a halo around the shell is decoration, and nothing in this interface
    // glows.
    const composer = new EffectComposer(renderer)
    composer.addPass(new RenderPass(scene, camera))
    composer.addPass(new OutputPass())

    const { front, back } = depthRange(position)
    handlesRef.current = {
      renderer,
      composer,
      camera,
      brainGroup,
      brainMesh,
      geometry,
      brainstemGeometry,
      material,
      lobeBoundaries,
      bandRings,
      nodeCloud: null,
      activeCloud: null,
      front,
      back,
      vertexBands: new Uint8Array(0),
      vertexBandCount: 0,
      baseColors: colors,
      sweep: 0,
      glowing: false,
    }

    const resize = () => {
      const width = container.clientWidth
      const height = container.clientHeight
      if (width === 0 || height === 0) return
      camera.aspect = width / height
      camera.updateProjectionMatrix()
      renderer.setSize(width, height)
      composer.setSize(width, height)
    }
    resize()
    const resizeObserver = new ResizeObserver(resize)
    resizeObserver.observe(container)

    // A slow turn is what makes a 3D form readable — the sway and the bob that
    // used to ride on top of it were float, not information, and are gone. The
    // turn itself stops entirely under `prefers-reduced-motion`, which leaves a
    // complete, static brain rather than a degraded one.
    const stillness = window.matchMedia('(prefers-reduced-motion: reduce)')

    let frameId: number
    const clock = new THREE.Clock()
    const animate = () => {
      const elapsed = clock.getElapsedTime()
      if (!userDriven) {
        brainGroup.rotation.y = stillness.matches ? 0.35 : elapsed * 0.04
      }

      applyActivity(handlesRef.current, activityRef.current)

      controls.update()
      composer.render()
      frameId = requestAnimationFrame(animate)
    }
    animate()

    return () => {
      cancelAnimationFrame(frameId)
      resizeObserver.disconnect()
      controls.dispose()
      container.removeChild(renderer.domElement)
      geometry.dispose()
      brainstemGeometry.dispose()
      material.dispose()
      brainstemMaterial.dispose()
      lobeBoundaries.children.forEach((line) => {
        ;(line as THREE.Line).geometry.dispose()
        ;((line as THREE.Line).material as THREE.Material).dispose()
      })
      disposeRings(bandRings)
      renderer.dispose()
      composer.dispose()
      handlesRef.current = null
    }
  }, [])

  // -- the node cloud: built once per atlas, never per selection ----------
  useEffect(() => {
    const handles = handlesRef.current
    if (!handles || !atlas) return

    const cloud = buildNodeCloud(atlas)
    handles.brainGroup.add(cloud)
    handles.nodeCloud = cloud

    return () => {
      handles.brainGroup.remove(cloud)
      cloud.geometry.dispose()
      ;(cloud.material as THREE.Material).dispose()
      handles.nodeCloud = null
    }
  }, [atlas])

  // -- the active cloud: rebuilt when the lit set changes ------------------
  useEffect(() => {
    const handles = handlesRef.current
    if (!handles) return
    // A version mismatch means these positions describe a different atlas, so
    // nothing is drawn rather than drawn in the wrong place.
    if (lit.nodes.length === 0 || !positionsUsable) return

    const cloud = buildActiveCloud(lit.nodes, lit.maxActivation)
    handles.brainGroup.add(cloud)
    handles.activeCloud = cloud

    // The idle sample steps back so the lit set reads as the data rather than
    // as a tint on the texture behind it.
    const idle = handles.nodeCloud?.material as THREE.PointsMaterial | undefined
    if (idle) idle.opacity = NODE_IDLE_OPACITY_LIT

    return () => {
      handles.brainGroup.remove(cloud)
      cloud.geometry.dispose()
      ;(cloud.material as THREE.Material).dispose()
      handles.activeCloud = null
      if (idle) idle.opacity = NODE_IDLE_OPACITY
    }
  }, [lit, positionsUsable])

  // -- picking: a 3-d tree over the active nodes, rebuilt on scope change --
  const tree = useMemo(
    () => (lit.nodes.length > 0 && positionsUsable ? new KdTree(lit.nodes) : null),
    [lit, positionsUsable],
  )

  useEffect(() => {
    const container = containerRef.current
    const handles = handlesRef.current
    if (!container || !handles || tree === null) {
      setHoveredNode(null)
      return
    }

    const raycaster = new THREE.Raycaster()
    const pointer = new THREE.Vector2()

    const onMove = (event: PointerEvent) => {
      const rect = container.getBoundingClientRect()
      pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1
      pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1
      raycaster.setFromCamera(pointer, handles.camera)

      // Into the brain group's own space, which is where node positions live.
      const inverse = new THREE.Matrix4().copy(handles.brainGroup.matrixWorld).invert()
      const origin = raycaster.ray.origin.clone().applyMatrix4(inverse)
      const target = raycaster.ray.origin
        .clone()
        .add(raycaster.ray.direction)
        .applyMatrix4(inverse)
      const direction = target.sub(origin).normalize()

      // Only the span where the cloud actually is. The ray starts at the
      // camera, several units away, so sampling from zero spends most of its
      // steps in empty space and leaves the steps that matter too far apart —
      // which is what made hovering miss: at 48 steps over 6 units the samples
      // sit 0.125 apart while the search radius was 0.05, so a node could fall
      // between two samples and never be found.
      const span = intersectSphere(origin, direction, SHELL_PICK_RADIUS)
      let found: LitNode | null = null
      if (span !== null) {
        const [near, far] = span
        // Radius comfortably over half the sample spacing, so consecutive
        // spheres overlap and the swept volume has no holes in it.
        const steps = PICK_STEPS
        const spacing = (far - near) / steps
        const radius = Math.max(PICK_RADIUS, spacing * 0.75)
        for (const point of rayPoints(origin, direction, near, far, steps)) {
          const hit = tree.nearest(point, radius)
          if (hit !== null) {
            found = hit
            break // frontmost along the ray
          }
        }
      }
      setHoveredNode(found)
    }

    const onLeave = () => setHoveredNode(null)
    container.addEventListener('pointermove', onMove)
    container.addEventListener('pointerleave', onLeave)
    return () => {
      container.removeEventListener('pointermove', onMove)
      container.removeEventListener('pointerleave', onLeave)
    }
  }, [tree])

  // -- paint: runs on every data change, rebuilds nothing expensive --------
  useEffect(() => {
    const handles = handlesRef.current
    if (!handles) return

    const { geometry, material, lobeBoundaries, bandRings, front, back } = handles
    const position = geometry.attributes.position as THREE.BufferAttribute
    const colorAttribute = geometry.attributes.color as THREE.BufferAttribute
    const bandCount = bandViews.length

    // Only when the band layout itself changed (a trace with a different layer
    // count), not when the selection moved.
    if (handles.vertexBandCount !== bandCount) {
      handles.vertexBands = assignVertexBands(position, front, back, bandCount)
      handles.vertexBandCount = bandCount
    }

    const tint = bandViews.map((view) => {
      const color = new THREE.Color()
      color.setHSL(view.color.h, view.color.s, view.color.l)
      // The band holding the selected layer is lifted so the shared selection
      // is visible on the brain, and the crossover band is lifted less.
      if (view.containsSelection) color.offsetHSL(0, 0.05, 0.14)
      return color
    })

    // Written into `baseColors` rather than straight onto the attribute: the
    // animation loop layers an activity glow on top of these every frame, so
    // it needs the unglowed values to start from.
    const base = handles.baseColors
    for (let i = 0; i < position.count; i++) {
      const color = tint[handles.vertexBands[i]] ?? tint[0]
      base[i * 3] = color.r
      base[i * 3 + 1] = color.g
      base[i * 3 + 2] = color.b
    }
    ;(colorAttribute.array as Float32Array).set(base)
    colorAttribute.needsUpdate = true
    handles.glowing = false

    // Bands need a touch more presence than the bare glass shell to read as
    // colour at all; with no trace the brain keeps its original translucency.
    material.opacity = lensAvailable ? 0.52 : 0.38

    // Data on screen means the anatomy recedes.
    const sulcusOpacity = lensAvailable ? 0.12 : 0.55
    lobeBoundaries.children.forEach((line) => {
      const lineMaterial = (line as THREE.Line).material as THREE.LineBasicMaterial
      lineMaterial.opacity = sulcusOpacity
    })

    disposeRings(bandRings)
    if (lensAvailable) {
      const span = front - back
      bandViews.forEach((view) => {
        // The ring at the band's leading edge; the last band's trailing edge is
        // the occipital pole, which has no ring to draw.
        if (view.band.index === 0) return
        const z = front - (view.band.index / bandCount) * span
        const points = ringPointsAtDepth(position, z)
        if (points === null) return
        const ringMaterial = new THREE.LineBasicMaterial({
          color: view.isCrossover ? 0xe6e8eb : 0x6e7681,
          transparent: true,
          opacity: view.isCrossover ? 0.95 : 0.28,
        })
        bandRings.add(
          new THREE.Line(new THREE.BufferGeometry().setFromPoints(points), ringMaterial),
        )
      })
    }
  }, [bandViews, lensAvailable])

  // -- hover: which band is the pointer over ------------------------------
  useEffect(() => {
    const container = containerRef.current
    const handles = handlesRef.current
    if (!container || !handles) return

    const raycaster = new THREE.Raycaster()
    const pointer = new THREE.Vector2()
    const local = new THREE.Vector3()

    const bandAtEvent = (event: PointerEvent | MouseEvent): number | null => {
      const rect = container.getBoundingClientRect()
      pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1
      pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1
      raycaster.setFromCamera(pointer, handles.camera)
      const hit = raycaster.intersectObject(handles.brainMesh, false)[0]
      if (!hit) return null
      // The group rotates, so the hit has to come back into the geometry's
      // own frame before its depth means anything.
      local.copy(hit.point)
      handles.brainMesh.worldToLocal(local)
      return bandForDepth(local.z, handles.front, handles.back, handles.vertexBandCount || 1)
    }

    const onMove = (event: PointerEvent) => setHoveredBand(bandAtEvent(event))
    const onLeave = () => setHoveredBand(null)
    const onClick = (event: MouseEvent) => {
      const index = bandAtEvent(event)
      if (index === null || !onSelectLayer) return
      const view = bandViews[index]
      if (view) onSelectLayer(view.band.startLayer)
    }

    container.addEventListener('pointermove', onMove)
    container.addEventListener('pointerleave', onLeave)
    container.addEventListener('click', onClick)
    return () => {
      container.removeEventListener('pointermove', onMove)
      container.removeEventListener('pointerleave', onLeave)
      container.removeEventListener('click', onClick)
    }
  }, [bandViews, onSelectLayer])

  const hovered = hoveredBand === null ? null : (bandViews[hoveredBand] ?? null)

  return (
    <div className="relative h-full w-full">
      <div className="h-full w-full" ref={containerRef} />

      <BrainLegend
        atlas={atlas}
        bandViews={bandViews}
        lensAvailable={lensAvailable}
        trace={trace}
        onSelectLayer={onSelectLayer}
      />

      <ActivityChip activity={activity} status={status} />

      <FeaturePanel
        agreement={agreement}
        atlas={atlas}
        lit={lit}
        onScopeChange={setScope}
        scope={scope}
        trace={trace}
      />

      {hoveredNode ? <NodeDetail node={hoveredNode} trace={trace} /> : null}

      {hovered ? <BandDetail view={hovered} /> : null}
    </div>
  )
}

function disposeRings(group: THREE.Group) {
  ;[...group.children].forEach((child) => {
    const line = child as THREE.Line
    line.geometry.dispose()
    ;(line.material as THREE.Material).dispose()
    group.remove(line)
  })
}

const CLASS_COPY: Record<string, string> = {
  answer: 'holds the final answer',
  echo: 'echoes the token here',
  other: 'neither',
}

/**
 * The legend, and the disclaimer that makes the whole view honest: a band is a
 * range of transformer layers, not a brain region. Nothing here names an
 * anatomical structure, because none of them correspond to anything the model
 * computes.
 */
function BrainLegend({
  atlas,
  bandViews,
  lensAvailable,
  trace,
  onSelectLayer,
}: {
  atlas: Atlas | null | undefined
  bandViews: BandView[]
  lensAvailable: boolean
  trace: Trace | null
  onSelectLayer?: (layer: number) => void
}) {
  return (
    <div className="pointer-events-none absolute top-3 left-3 max-w-[16rem] space-y-2 text-[11px]">
      <AtlasNote atlas={atlas} />
      <p className="text-text-tertiary font-medium tracking-[0.04em] uppercase">Layer bands</p>

      {trace === null ? (
        <p className="text-text-secondary leading-4">
          Run a prompt to paint this brain with the model&apos;s layer-by-layer readouts.
        </p>
      ) : !lensAvailable ? (
        <p className="text-const leading-4">
          This trace has no logit-lens data, so the bands are unpainted. The lens pass did not
          run for it.
        </p>
      ) : (
        <ul className="divide-border-subtle border-border-subtle bg-bg-elevated pointer-events-auto divide-y rounded-[10px] border">
          {bandViews.map((view) => (
            <li key={view.band.index}>
              <button
                className={`flex w-full items-center gap-2 px-2 py-1 text-left transition-colors duration-150 ${
                  view.containsSelection
                    ? 'border-l-2 border-l-fn bg-fn/[0.06] pl-1.5'
                    : 'hover:bg-white/[0.02]'
                }`}
                onClick={() => onSelectLayer?.(view.band.startLayer)}
                type="button"
              >
                {/* The colour never stands alone: the class it encodes is
                    spelled out in the same row. */}
                <span
                  aria-hidden="true"
                  className="size-2 shrink-0 rounded-full"
                  style={{ backgroundColor: cssColor(view.color) }}
                />
                <span className="text-text-primary font-mono tabular-nums">
                  L{view.band.startLayer}–{view.band.endLayer}
                </span>
                <span className="text-text-tertiary truncate">
                  {view.state?.klass ? CLASS_COPY[view.state.klass] : 'no data'}
                </span>
                {view.isCrossover ? (
                  <span className="text-text-primary ml-auto shrink-0 font-mono">settles</span>
                ) : null}
              </button>
            </li>
          ))}
        </ul>
      )}

      <p className="text-text-tertiary leading-4">
        A band is a range of transformer layers, front to back. It is not a brain region — no
        anatomical area computes any of this.
      </p>
    </div>
  )
}

/**
 * What the node cloud is, and what it is not allowed to claim.
 *
 * Every clause here is load-bearing. A node is a feature, not a place. Nearby
 * nodes really are similar and distant ones mean nothing, because UMAP
 * preserves neighbourhoods and distorts global distance — so the view offers
 * no axes, no coordinates and no scale, and the copy has to say why rather
 * than leave the absence to be noticed. And the sample is stated as a sample:
 * this file carries 20,000 of the atlas's features so the brain has structure
 * before a prompt, and it is not the set a trace's activations are read
 * against.
 */
function AtlasNote({ atlas }: { atlas: Atlas | null | undefined }) {
  if (atlas === undefined) {
    return <p className="text-text-tertiary leading-4">Loading the feature atlas…</p>
  }

  if (atlas === null) {
    return (
      <div className="space-y-1">
        <p className="text-text-tertiary font-medium tracking-[0.04em] uppercase">Feature nodes</p>
        <p className="text-const leading-4">
          No feature atlas is available, so no nodes are drawn. Build one with{' '}
          <span className="text-text-primary font-mono">scripts/build_feature_atlas.py</span>.
        </p>
      </div>
    )
  }

  const layers = atlasLayers(atlas)
  const named = namedAreaCount(atlas)

  return (
    <div className="space-y-1">
      <p className="text-text-tertiary font-medium tracking-[0.04em] uppercase">Feature nodes</p>
      <p className="text-text-secondary leading-4">
        {atlas.nodes.length.toLocaleString()} of {atlas.total.toLocaleString()} SAE features,
        sampled — one dot per feature, across layers {layers.join(', ')}.
      </p>

      {/* What "near" means, which depends on the atlas's source. Stated rather
          than left to be inferred: the two sources support different claims and
          reading one as the other is the mistake worth preventing. */}
      <p className="text-text-tertiary leading-4">
        <span className="text-text-secondary">{sourceClaim(atlas.source)}</span>.{' '}
        {atlas.source === 'labels'
          ? 'That is a map of how features were described, not of what the model computes.'
          : 'That is the model’s own geometry.'}
      </p>

      {/* The layout's fidelity as a number. A picture nobody measured is
          decoration, so the measurement is on screen rather than in a log. */}
      {atlas.knnPreservation !== null ? (
        <p className="text-text-tertiary leading-4">
          Of a feature&apos;s {atlas.knnK ?? 20} nearest neighbours,{' '}
          <span className="text-text-primary font-mono tabular-nums">
            {(atlas.knnPreservation * 100).toFixed(0)}%
          </span>{' '}
          survive the flattening to three dimensions. Distance beyond a
          neighbourhood is meaningless, which is why there is no scale to read.
        </p>
      ) : (
        <p className="text-text-tertiary leading-4">
          This atlas records no fidelity measurement, so how much of the original
          structure survived is unknown.
        </p>
      )}

      <p className="text-text-tertiary leading-4">
        {atlas.areas.length} clusters, {named === 0 ? 'none named' : `${named} named`} — a cluster
        earns a name only when its features&apos; own labels agree more than a random group of the
        same size.
        {atlas.explainerAmi !== null && atlas.source === 'labels' ? (
          <>
            {' '}
            Explainer influence{' '}
            <span className="text-text-primary font-mono tabular-nums">
              {atlas.explainerAmi.toFixed(3)}
            </span>
            : the
            areas are not an artifact of which model wrote the labels.
          </>
        ) : null}
      </p>

      <p className="text-text-tertiary leading-4">
        A dot is a feature, not a place. No named brain region computes any of this.
      </p>
      <p className="text-text-disabled font-mono text-[10px] leading-4">
        atlas {atlas.version} · {atlas.source}
      </p>
    </div>
  )
}

/**
 * The per-band breakdown. A blended band's colour is one class and one number
 * standing in for several layers, so the layers it was blended from are always
 * available rather than the colour being the only story.
 */
const SCOPE_COPY: Record<LitScope, { label: string; says: string }> = {
  cell: { label: 'Cell', says: 'one layer at one token' },
  token: { label: 'Token', says: 'every layer at one token' },
  trace: { label: 'Trace', says: 'every layer at every token' },
}

/**
 * What is lit, at what scope, and everything that is true of the set but not
 * visible in the picture.
 *
 * The picture cannot say any of this on its own. A cloud of bright dots looks
 * identical whether it is the complete set or a sixteenth of it, whether the
 * missing layers were empty or never computed, and whether the positions
 * belong to this atlas or another one. So each of those is written down.
 */
function FeaturePanel({
  agreement,
  atlas,
  lit,
  onScopeChange,
  scope,
  trace,
}: {
  agreement: ReturnType<typeof atlasAgreement>
  atlas: Atlas | null | undefined
  lit: LitSet
  onScopeChange: (scope: LitScope) => void
  scope: LitScope
  trace: Trace | null
}) {
  if (trace === null) return null

  const coverage = coverageNote(lit, atlas)

  return (
    <div className="pointer-events-none absolute top-3 right-3 max-w-[17rem] space-y-2 text-[11px]">
      <p className="text-text-tertiary font-medium tracking-[0.04em] uppercase">Lit features</p>

      {/* 4.6 — the scope, and what it means, stated rather than implied. */}
      <div className="border-border-subtle bg-bg-elevated pointer-events-auto flex rounded-[10px] border">
        {(['cell', 'token', 'trace'] as const).map((option) => (
          <button
            className={`flex-1 px-2 py-1 font-mono transition-colors duration-150 first:rounded-l-[9px] last:rounded-r-[9px] ${
              option === scope
                ? 'bg-fn/[0.10] text-text-primary'
                : 'text-text-tertiary hover:bg-white/[0.02]'
            }`}
            key={option}
            onClick={() => onScopeChange(option)}
            type="button"
          >
            {SCOPE_COPY[option].label}
          </button>
        ))}
      </div>
      <p className="text-text-tertiary leading-4">{SCOPE_COPY[scope].says}</p>

      {/* 4.5 — a mismatch is stated before any count, because if it holds the
          counts describe a different map. */}
      {agreement.kind === 'mismatch' ? (
        <p className="text-const leading-4">
          This trace was placed against atlas{' '}
          <span className="font-mono">{agreement.traceVersion}</span>, but the atlas loaded here
          is <span className="font-mono">{agreement.atlasVersion}</span>. Its positions are not
          shown, because they do not correspond to this map.
        </p>
      ) : agreement.kind === 'no-atlas' ? (
        <p className="text-const leading-4">{agreement.reason}</p>
      ) : null}

      {/* 4.10 — nothing lit, and why. Never a silent empty view. */}
      {lit.nodes.length === 0 && lit.unplaced.length === 0 ? (
        <p className="text-const leading-4">
          {lit.emptyReason ?? 'nothing lit at this scope'}
        </p>
      ) : (
        <dl className="text-text-secondary space-y-0.5 leading-4">
          <div className="flex justify-between gap-2">
            <dt>drawn</dt>
            <dd className="text-text-primary font-mono tabular-nums">{lit.nodes.length}</dd>
          </div>
          {/* 4.8 — the slice never reads as the whole. */}
          <div className="flex justify-between gap-2">
            <dt>shown of fired</dt>
            <dd className="text-text-primary font-mono tabular-nums">
              {lit.shown}
              {lit.fired === null ? '' : ` / ${lit.fired}`}
            </dd>
          </div>
          {lit.aggregation !== null ? (
            <div className="flex justify-between gap-2">
              <dt>combined by</dt>
              <dd className="text-text-primary font-mono">{AGGREGATION}</dd>
            </div>
          ) : null}
        </dl>
      )}

      {lit.fired !== null && lit.shown < lit.fired ? (
        <p className="text-text-tertiary leading-4">
          The SAE pass keeps the strongest features per cell, not all of them — {lit.shown} of{' '}
          {lit.fired} that fired.
        </p>
      ) : null}

      {/* 4.4 — a feature that fired but has nowhere honest to be drawn. */}
      {lit.unplaced.length > 0 ? (
        <div className="space-y-1">
          <p className="text-const leading-4">
            {lit.unplaced.length} feature{lit.unplaced.length === 1 ? '' : 's'} the atlas cannot
            place {lit.unplaced.length === 1 ? 'is' : 'are'} not drawn — no position is invented
            for {lit.unplaced.length === 1 ? 'it' : 'them'}.
          </p>
          <ul className="border-border-subtle bg-bg-elevated max-h-24 overflow-y-auto rounded-[10px] border">
            {lit.unplaced.slice(0, 24).map((feature) => (
              <li
                className="flex justify-between gap-2 px-2 py-0.5 font-mono tabular-nums"
                key={`${feature.layer}/${feature.feature}`}
              >
                <span className="text-text-secondary">
                  L{feature.layer} #{feature.feature}
                </span>
                <span className="text-text-tertiary">{feature.activation.toFixed(2)}</span>
              </li>
            ))}
          </ul>
        </div>
      ) : null}

      {/* 4.10 — a layer with no data, named as such. */}
      {coverage ? <p className="text-text-tertiary leading-4">{coverage}</p> : null}

      {/* 4.7 — the exclusion, and the reason for it. */}
      {lit.bosExcluded ? <p className="text-text-tertiary leading-4">{BOS_REASON}.</p> : null}
    </div>
  )
}

/**
 * One node, under the pointer.
 *
 * A feature with no explanation shows the absence rather than an empty line:
 * Neuronpedia has no label for some features, and a blank where text belongs
 * reads as a loading state or a bug instead of as the fact it is.
 */
function NodeDetail({ node, trace }: { node: LitNode; trace: Trace | null }) {
  const key = `${node.layer}/${node.feature}`
  const label = trace?.labels[key] ?? null
  const record = trace?.passes.find((p) => p.name === 'labels')
  const labelsRan = record !== undefined

  return (
    <div className="border-border-subtle bg-bg-elevated absolute bottom-3 left-3 max-w-[20rem] space-y-1 rounded-[10px] border px-2.5 py-2 text-[11px]">
      <p className="text-text-primary font-mono tabular-nums">
        L{node.layer} #{node.feature}
        <span className="text-text-tertiary"> · act {node.activation.toFixed(2)}</span>
        <span className="text-text-tertiary">
          {' '}
          · {node.cluster === -1 ? 'no area' : `area ${node.cluster}`}
        </span>
      </p>

      {label ? (
        <p className="text-text-secondary leading-4">{label.text.trim()}</p>
      ) : labelsRan ? (
        <p className="text-const leading-4">
          Neuronpedia has no explanation for this feature.
        </p>
      ) : (
        <p className="text-const leading-4">
          No labels on this trace — it was run without the labels pass.
        </p>
      )}

      <a
        className="text-fn block font-mono hover:underline"
        href={`https://www.neuronpedia.org/gemma-2-2b/${node.layer}-gemmascope-res-16k/${node.feature}`}
        rel="noreferrer"
        target="_blank"
      >
        neuronpedia ↗
      </a>
    </div>
  )
}

function BandDetail({ view }: { view: BandView }) {
  return (
    <div className="border-border-strong bg-bg-elevated pointer-events-none absolute right-3 bottom-3 max-w-[16rem] rounded-[10px] border p-2.5 text-[11px]">
      <p className="text-text-primary font-mono tabular-nums">
        layers {view.band.startLayer}–{view.band.endLayer}
        {view.isCrossover ? (
          <span className="text-text-primary ml-2">· settles here</span>
        ) : null}
      </p>

      {view.state === null || view.state.klass === null ? (
        <p className="text-text-tertiary mt-1">No logit-lens readout for these layers.</p>
      ) : (
        <>
          <p className="text-text-secondary mt-1">
            blended as <span className="text-text-primary">{view.state.klass}</span> at{' '}
            {(view.state.confidence * 100).toFixed(0)}% — from {view.state.counts.answer} answer,{' '}
            {view.state.counts.echo} echo, {view.state.counts.other} other
          </p>
          <ol className="mt-1.5 space-y-0.5">
            {view.state.layers.map((layer) => (
              <li className="flex items-center gap-2 font-mono" key={layer.layer}>
                <span className="text-text-tertiary w-8 tabular-nums">L{layer.layer}</span>
                <span
                  aria-hidden="true"
                  className="size-2 shrink-0 rounded-full"
                  style={{ backgroundColor: cssColor(classColor(layer.klass, layer.confidence)) }}
                />
                <span className="text-text-primary flex-1">{layer.klass}</span>
                <span className="text-text-secondary tabular-nums">
                  {(layer.confidence * 100).toFixed(0)}%
                </span>
              </li>
            ))}
          </ol>
          {view.state.undecoded.length > 0 ? (
            <p className="text-text-tertiary mt-1.5">
              not decoded: {view.state.undecoded.map((l) => `L${l}`).join(', ')}
            </p>
          ) : null}
        </>
      )}
    </div>
  )
}

// --------------------------------------------------------------------------
// activity rendering
// --------------------------------------------------------------------------

/** How fast the sweep closes on the last reported band, per frame. */
const SWEEP_EASE = 0.1
/** Width of the sweep's glow, in band units. */
const SWEEP_WIDTH = 0.85

/**
 * Layer the current activity onto the painted band colours.
 *
 * Two rules make this honest. The sweep eases *toward* the last reported band
 * and is clamped there, so the brain can lag the model but never runs ahead of
 * it. And only the lens phase gets a positional glow at all — every other
 * working state gets one undifferentiated pulse across the whole shell, which
 * says "busy" without naming a layer.
 */
function applyActivity(handles: SceneHandles | null, activity: Activity): void {
  if (!handles) return
  const colorAttribute = handles.geometry.attributes.color as THREE.BufferAttribute
  const target = colorAttribute.array as Float32Array
  const base = handles.baseColors

  if (activity.kind === 'idle') {
    // Nothing running: restore the painted colours once, then leave the buffer
    // alone so an idle brain costs no per-frame writes.
    if (handles.glowing) {
      target.set(base)
      colorAttribute.needsUpdate = true
      handles.glowing = false
    }
    return
  }

  if (activity.kind === 'lens') {
    // Approach-only: `sweep` never exceeds the band the service last reported.
    handles.sweep = Math.min(
      activity.targetBand,
      handles.sweep + (activity.targetBand - handles.sweep) * SWEEP_EASE,
    )
  }

  // A steady lift, not a breathing one. The old sine pulse was a glow that
  // loops forever, which says nothing the word beside it does not already say.
  const pulse = 0.14
  const positional = activity.kind === 'lens'

  for (let i = 0; i < handles.vertexBands.length; i++) {
    const band = handles.vertexBands[i]
    const glow = positional
      ? 0.9 * Math.exp(-(((band - handles.sweep) / SWEEP_WIDTH) ** 2))
      : pulse
    const offset = i * 3
    target[offset] = Math.min(1, base[offset] + glow)
    target[offset + 1] = Math.min(1, base[offset + 1] + glow)
    target[offset + 2] = Math.min(1, base[offset + 2] + glow)
  }
  colorAttribute.needsUpdate = true
  handles.glowing = true
}

/**
 * The one-line answer to "what is it doing", in the units the service actually
 * reported. Never names a layer during generation.
 */
function ActivityChip({ activity, status }: { activity: Activity; status: RunState }) {
  if (activity.kind === 'idle') return null

  const copy =
    activity.kind === 'generating'
      ? `generating · token ${activity.done} of ${activity.total}`
      : activity.kind === 'lens'
        ? `reading every layer · ${activity.done} of ${activity.total}`
        : status === 'warming'
          ? 'loading the model…'
          : status === 'pending'
            ? 'queued…'
            : 'working…'

  return (
    // A 6px dot plus the words, in the palette's own status colour — never a
    // filled pill, and never a pulse: the words already say it is working.
    <div
      aria-live="polite"
      className="border-border-strong bg-bg-elevated text-text-secondary pointer-events-none absolute top-3 right-3 rounded-sm border px-2.5 py-1 font-mono text-[10px] tabular-nums"
    >
      <span aria-hidden="true" className="bg-fn mr-1.5 inline-block size-1.5 rounded-full" />
      {copy}
    </div>
  )
}

import { useEffect, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js'
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js'
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'

import type { JobProgress, Trace } from '@/lib/api-types'
import type { Atlas, AtlasArea } from '@/lib/atlas'
import {
  atlasLayers,
  loadAtlas,
  namedAreaCount,
  positionBuffer,
  sourceClaim,
} from '@/lib/atlas'
import type { LabelFilter, LitArea, LitNode, LitScope, LitSet } from '@/lib/lit'
import {
  AGGREGATION,
  AREA_AGGREGATION,
  BOS_REASON,
  atlasAgreement,
  coverageNote,
  filterByLabel,
  litAreas,
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
 * design.md). `layers` is a phase that counts layers — the SAE pass, or the
 * lens pass — and it reports its position in that count without the shell
 * claiming to show where it is: depth is not a spatial axis here, so there is
 * nowhere on the shell for a layer to be.
 */
type ActivityKind = 'idle' | 'unknown' | 'generating' | 'layers'

interface Activity {
  kind: ActivityKind
  done: number
  total: number
}

const IDLE: Activity = { kind: 'idle', done: 0, total: 0 }

/**
 * What moved the shared selection.
 *
 * The brain needs this, not just the new value: "the user clicked one cell"
 * and "a new trace defaulted the selection" mean different things about what
 * should be lit, and a bare (layer, token) pair cannot tell them apart.
 */
export type SelectionVia = 'cell' | 'token' | 'layer' | 'default'

export interface BrainProps {
  trace: Trace | null
  /** The (layer, token) the grid and the brain share. */
  selection: { layer: number; position: number; via?: SelectionVia } | null
  status: RunState
  progress: JobProgress | null
  /** The transport steps the shared selection's layer. */
  onSelectLayer?: (layer: number) => void
}

/**
 * The shell's own tone: the interface's primary #E6E8EB, never pure white.
 * One flat colour, because nothing is painted onto the shell any more — the
 * data is the cloud inside it.
 */
const SHELL_COLOR = new THREE.Color(0xe6e8eb)

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

// --------------------------------------------------------------------------
// areas: a cluster of the atlas, drawn as a soft mass rather than an outline
// --------------------------------------------------------------------------

/**
 * Why a glow and not a hull.
 *
 * An outline says "the boundary is here", and a UMAP cluster has no boundary
 * worth drawing — the projection distorts distance past a neighbourhood, so
 * the edge of a convex hull would be an artifact of the flattening presented
 * as a fact about the model. A soft mass at the centroid claims only what the
 * atlas measured: something is concentrated around here.
 *
 * Size comes from the cluster's own recorded `spread`, so a diffuse area looks
 * diffuse, and brightness from its members' activations — never the reverse.
 */
const AREA_COLOR = new THREE.Color(0x7ee787)
const AREA_MIN_SIZE = 0.34
const AREA_MAX_SIZE = 1.5

const AREA_VERTEX_SHADER = `
attribute float prominence;
attribute float spread;
uniform float detail;
varying float vProminence;
void main() {
  vProminence = prominence;
  vec4 viewPosition = modelViewMatrix * vec4(position, 1.0);
  float size = clamp(spread * 14.0, ${AREA_MIN_SIZE.toFixed(2)}, ${AREA_MAX_SIZE.toFixed(2)});
  // Areas recede as the camera closes in: up close the nodes are the subject
  // and a soft mass over them is fog.
  gl_PointSize = size * mix(1.0, 0.45, detail) * (300.0 / -viewPosition.z);
  gl_Position = projectionMatrix * viewPosition;
}
`

const AREA_FRAGMENT_SHADER = `
uniform vec3 color;
uniform float detail;
varying float vProminence;
void main() {
  vec2 offset = gl_PointCoord - vec2(0.5);
  float radius = length(offset);
  if (radius > 0.5) discard;
  // A wide gaussian falloff: no edge anywhere, because there is no boundary
  // to claim.
  float falloff = exp(-radius * radius * 11.0) - exp(-2.75);
  gl_FragColor = vec4(color, falloff * mix(0.30, 0.10, detail) * mix(0.25, 1.0, vProminence));
}
`

function buildAreaCloud(areas: LitArea[]): THREE.Points {
  const positions = new Float32Array(areas.length * 3)
  const prominences = new Float32Array(areas.length)
  const spreads = new Float32Array(areas.length)
  for (let i = 0; i < areas.length; i++) {
    const [x, y, z] = areas[i].area.centroid
    positions[i * 3] = x
    positions[i * 3 + 1] = y
    positions[i * 3 + 2] = z
    prominences[i] = areas[i].prominence
    spreads[i] = areas[i].area.spread
  }

  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
  geometry.setAttribute('prominence', new THREE.BufferAttribute(prominences, 1))
  geometry.setAttribute('spread', new THREE.BufferAttribute(spreads, 1))

  const material = new THREE.ShaderMaterial({
    uniforms: { color: { value: AREA_COLOR }, detail: { value: 0 } },
    vertexShader: AREA_VERTEX_SHADER,
    fragmentShader: AREA_FRAGMENT_SHADER,
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  })

  const points = new THREE.Points(geometry, material)
  points.frustumCulled = false
  return points
}

// --------------------------------------------------------------------------
// level of detail
// --------------------------------------------------------------------------

/**
 * How close the camera is, 0 (furthest out) to 1 (closest in), against the
 * orbit control's own limits.
 *
 * One number drives the whole progressive disclosure: far out, areas carry
 * the view and no single node's identity is needed to read it; close in, the
 * area masses fade back and the nodes grow into things you can aim at.
 */
function detailLevel(distance: number, near: number, far: number): number {
  if (far <= near) return 0
  const t = (far - distance) / (far - near)
  return t < 0 ? 0 : t > 1 ? 1 : t
}

/** Clearance between two area labels before one of them gives way, in px. */
const LABEL_GAP = 10

/**
 * Where the area labels give way to the nodes.
 *
 * Past the default framing, not at it: the brain opens at roughly 0.62 on
 * this scale, and a window that began before that would hand the viewer a
 * half-faded overview they never asked to leave.
 */
const LABEL_FADE_START = 0.70
const LABEL_FADE_END = 0.93

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
uniform float detail;
varying float vProminence;
void main() {
  vProminence = prominence;
  vec4 viewPosition = modelViewMatrix * vec4(position, 1.0);
  // sizeAttenuation, by hand: nodes shrink with distance so the cloud reads
  // as a volume rather than a flat spray. The detail uniform grows them
  // further as the camera closes in, so an approached node is aimable.
  gl_PointSize = mix(${ACTIVE_MIN_SIZE.toFixed(3)}, ${ACTIVE_MAX_SIZE.toFixed(3)}, prominence)
    * mix(0.85, 1.45, detail)
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
    uniforms: { color: { value: ACTIVE_COLOR }, detail: { value: 0 } },
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
  /** The atlas's node cloud, once an atlas has loaded. Null until then. */
  nodeCloud: THREE.Points | null
  /** The features the current trace lit, at the current scope. */
  activeCloud: THREE.Points | null
  /** The lit areas, as soft masses at their centroids. */
  areaCloud: THREE.Points | null
  /**
   * One absolutely-positioned label per drawn area, projected from its
   * centroid every frame.
   *
   * DOM rather than a texture in the scene: the names are real text that has
   * to stay crisp at any zoom, wrap, and be selectable — a canvas-drawn
   * string is none of those. They are moved imperatively from the animation
   * loop, never through React, so a rotating brain does not re-render the
   * component 60 times a second.
   */
  areaLabels: {
    element: HTMLDivElement
    centroid: THREE.Vector3
    /** Whether this trace lit anything inside the area — it wins collisions. */
    lit: boolean
    /** Measured once at creation; the text never changes after that. */
    width: number
    height: number
  }[]
  /** The lift currently applied to the shell, so an idle frame writes nothing. */
  lift: number
  /** 0 (furthest) to 1 (closest), recomputed each frame from the camera. */
  detail: number
  /** Orbit limits, for turning camera distance into `detail`. */
  minDistance: number
  maxDistance: number
}

export function Brain({ trace, selection, status, progress, onSelectLayer }: BrainProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const handlesRef = useRef<SceneHandles | null>(null)

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

  // Which area's label the pointer is over. Carries the atlas's record and
  // what this trace lit inside it, because the disclosure needs both.
  const [hoveredArea, setHoveredArea] = useState<{
    area: AtlasArea
    lit: LitArea | null
  } | null>(null)

  // The label-text filter over the lit set. Empty means no filter at all,
  // which is a different state from "a query that matched nothing".
  const [query, setQuery] = useState('')

  const labelsRef = useRef<HTMLDivElement>(null)

  // The features this trace lit. Positions come from `trace.layout`, never
  // from the atlas asset — see litSet's own note on why that distinction is
  // not cosmetic.
  const lit = useMemo<LitSet>(() => litSet(trace, scope, selection), [trace, scope, selection])

  // The filter narrows what is drawn, so everything downstream — the cloud,
  // the areas, the picking tree — reads the filtered set rather than the full
  // one. A filtered view that still glowed with filtered-out features would
  // be lying about what it is showing.
  const filter = useMemo<LabelFilter>(() => filterByLabel(lit, trace, query), [lit, trace, query])
  const drawn = filter.nodes

  // Areas are aggregated from the members actually on screen.
  const areas = useMemo<LitArea[]>(
    () => litAreas({ ...lit, nodes: drawn }, atlas),
    [lit, drawn, atlas],
  )

  // Whether the trace's positions correspond to the atlas on screen. A
  // mismatch is not a warning to bury: the areas, names and neighbourhoods on
  // screen belong to a different map, so the positions may not be presented
  // as this atlas's.
  const agreement = useMemo(() => atlasAgreement(trace, atlas), [trace, atlas])
  const positionsUsable = agreement.kind === 'match' || agreement.kind === 'no-view-atlas'

  // Read by the animation loop every frame. A ref rather than state so a new
  // progress reading never restarts the loop or rebuilds the scene.
  const activityRef = useRef<Activity>(IDLE)

  // 6.5 — the selection decides what is lit whenever the selection is what
  // changed. Clicking one grid cell means "this cell", so the brain narrows
  // to it; picking a token means that token's whole column; the transport
  // walks depth, which is only depth-shaped at cell scope; and a new trace
  // returns to the documented default. Choosing a scope by hand still holds
  // until the next selection arrives.
  //
  // Adjusted during render rather than in an effect: this is state derived
  // from a prop change, so an effect would commit the old scope, paint it,
  // and then correct itself — one frame of the wrong set lit.
  const via = selection?.via
  const moved = `${via ?? ''}:${selection?.layer ?? ''}:${selection?.position ?? ''}`
  const [lastMoved, setLastMoved] = useState(moved)
  if (moved !== lastMoved) {
    setLastMoved(moved)
    if (via === 'cell' || via === 'layer') setScope('cell')
    else if (via === 'token') setScope('token')
    else if (via === 'default') setScope('trace')
  }

  // Which layers the SAE pass actually ran on, for the whole trace.
  //
  // Deliberately not `lit.layersWithData`: at cell scope that set describes
  // the one selected cell, so stepping onto a layer with no features would
  // report the whole trace as featureless and disable the very control that
  // steps back off it.
  const featureLayers = useMemo(() => {
    if (!trace) return []
    const has = new Set<number>()
    for (const step of trace.steps) {
      for (const state of step.layers) if (state.features.length > 0) has.add(state.layer)
    }
    return [...has].sort((a, b) => a - b)
  }, [trace])

  // The activity the brain is entitled to show, derived from the job's own
  // reported state. It counts, and never places: a layer counter is a number,
  // not a position on the shell.
  const activity = useMemo<Activity>(() => {
    if (status === 'error' || status === 'idle' || status === 'done') return IDLE
    // Submitted, or warming up, and nothing reported yet.
    if (status === 'warming' || status === 'pending' || progress === null) {
      return { kind: 'unknown', done: 0, total: 0 }
    }
    if (progress.phase === 'generating') {
      return { kind: 'generating', done: progress.done, total: progress.total }
    }
    // Every other phase counts layers.
    return { kind: 'layers', done: progress.done, total: progress.total }
  }, [status, progress])

  useEffect(() => {
    activityRef.current = activity
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

    // Translucent glass shell — see-through so the node cloud inside it, and
    // the fold structure behind that, read clearly against black. One flat
    // tone: nothing is painted onto the shell, because nothing the model does
    // happens at a place on it.
    const material = new THREE.MeshPhysicalMaterial({
      color: SHELL_COLOR.clone(),
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
    // opacity whenever nodes are on screen, so nobody reads a named anatomical
    // region as the site of a feature.
    const lobeBoundaries = buildLobeBoundaries()
    brainGroup.add(lobeBoundaries)

    // Brainstem: a short tapered stalk beneath the cerebellum, so the
    // silhouette reads as a brain rather than a bare cerebral mass. Its own
    // material, so the working lift on the shell does not ride on the stem.
    const brainstemGeometry = new THREE.CylinderGeometry(0.075, 0.12, 0.4, 16)
    const brainstemMaterial = material.clone()
    brainstemMaterial.color = SHELL_COLOR.clone()
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
      nodeCloud: null,
      activeCloud: null,
      areaCloud: null,
      areaLabels: [],
      lift: 0,
      detail: 0,
      minDistance: controls.minDistance,
      maxDistance: controls.maxDistance,
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
      applyDetail(handlesRef.current)

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
    if (drawn.length === 0 || !positionsUsable) return

    const cloud = buildActiveCloud(drawn, lit.maxActivation)
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
  }, [drawn, lit.maxActivation, positionsUsable])

  // -- the area masses: rebuilt when what is lit inside them changes ------
  useEffect(() => {
    const handles = handlesRef.current
    if (!handles || areas.length === 0 || !positionsUsable) return

    const cloud = buildAreaCloud(areas)
    // Behind the nodes in draw order, so a lit node always reads over the
    // mass it belongs to rather than being washed out by it.
    handles.brainGroup.add(cloud)
    handles.areaCloud = cloud

    return () => {
      handles.brainGroup.remove(cloud)
      cloud.geometry.dispose()
      ;(cloud.material as THREE.Material).dispose()
      handles.areaCloud = null
    }
  }, [areas, positionsUsable])

  // -- the area labels: DOM, moved by the loop, rebuilt when the set changes
  //
  // A named area always carries its name, trace or no trace — that is what
  // makes the idle brain legible. An unnamed one gets a label only when it is
  // lit, so a glowing mass is never unexplained, and its text says only that
  // it is unnamed: the atlas measured its members' labels and found no
  // agreement to name it by, and inventing a summary here would be exactly
  // the claim that measurement refused to make.
  const labelled = useMemo(() => {
    if (!atlas) return []
    const byCluster = new Map(areas.map((entry) => [entry.area.cluster, entry]))
    return atlas.areas
      .filter((area) => area.name !== null || byCluster.has(area.cluster))
      .map((area) => ({ area, lit: byCluster.get(area.cluster) ?? null }))
  }, [atlas, areas])

  useEffect(() => {
    const handles = handlesRef.current
    const container = labelsRef.current
    if (!handles || !container) return

    const made = labelled.map(({ area, lit: inside }) => {
      const element = document.createElement('div')
      element.className =
        'pointer-events-auto absolute top-0 left-0 max-w-[9rem] cursor-default text-center ' +
        'font-mono text-[10px] leading-3 whitespace-normal ' +
        (area.name === null ? 'text-text-disabled italic' : 'text-text-primary')
      // The shell behind a name is bright and uneven, so the contrast has to
      // travel with the text rather than depend on what it happens to be over.
      element.style.textShadow = '0 1px 2px rgba(6,7,9,0.95), 0 0 6px rgba(6,7,9,0.8)'
      element.style.willChange = 'transform, opacity'
      element.style.visibility = 'hidden'
      element.textContent = area.name ?? 'unnamed area'
      element.addEventListener('pointerenter', () => setHoveredArea({ area, lit: inside }))
      element.addEventListener('pointerleave', () => setHoveredArea(null))
      container.appendChild(element)
      // Measured now, while it is in the document: a hidden element still has
      // a box, and the text never changes after this.
      const box = element.getBoundingClientRect()
      return {
        element,
        centroid: new THREE.Vector3(...area.centroid),
        lit: inside !== null,
        width: box.width,
        height: box.height,
      }
    })
    handles.areaLabels = made

    return () => {
      made.forEach(({ element }) => element.remove())
      handles.areaLabels = []
      setHoveredArea(null)
    }
  }, [labelled])

  // -- picking: a 3-d tree over the active nodes, rebuilt on scope change --
  const tree = useMemo(
    () => (drawn.length > 0 && positionsUsable ? new KdTree(drawn) : null),
    [drawn, positionsUsable],
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

  // -- the shell recedes once there is data inside it ---------------------
  useEffect(() => {
    const handles = handlesRef.current
    if (!handles) return

    // Data on screen means the anatomy is silhouette. With no atlas there is
    // nothing but the shell, so the folds are all there is to look at and
    // they stay legible.
    const sulcusOpacity = !atlas ? 0.55 : lit.nodes.length > 0 ? 0.12 : 0.2
    handles.lobeBoundaries.children.forEach((line) => {
      const lineMaterial = (line as THREE.Line).material as THREE.LineBasicMaterial
      lineMaterial.opacity = sulcusOpacity
    })
  }, [atlas, lit])

  return (
    <div className="relative h-full w-full">
      <div className="h-full w-full" ref={containerRef} />

      {/* Area names, over the canvas and moved by the animation loop. */}
      <div className="pointer-events-none absolute inset-0 overflow-hidden" ref={labelsRef} />

      <BrainLegend atlas={atlas} />

      <ActivityChip activity={activity} status={status} />
      <ComputingNote activity={activity} />

      <FeaturePanel
        agreement={agreement}
        areas={areas}
        atlas={atlas}
        featureLayers={featureLayers}
        filter={filter}
        lit={lit}
        onQueryChange={setQuery}
        onScopeChange={setScope}
        onSelectLayer={onSelectLayer}
        query={query}
        scope={scope}
        selection={selection}
        trace={trace}
      />

      {/* One detail panel, and the area wins: the pointer is over its label,
          which sits above the cloud, so a node behind it is not what is being
          asked about. */}
      {hoveredArea ? (
        <AreaDetail entry={hoveredArea} />
      ) : hoveredNode ? (
        <NodeDetail node={hoveredNode} trace={trace} />
      ) : null}
    </div>
  )
}

/**
 * The legend, and the disclaimers that make the whole view honest.
 *
 * Every clause is load-bearing. A node is one SAE feature, not a place. An
 * area is a cluster of features whose directions in the residual stream are
 * close together, not a region of a brain — and a blob on a brain is a very
 * strong invitation to read it as one, so the denial is on screen rather than
 * in a doc. Nearby nodes really are similar and distant ones mean nothing,
 * because the projection preserves neighbourhoods and distorts global
 * distance: hence no axes, no coordinates, no scale, and copy that says why
 * instead of leaving the absence to be noticed. And the sample is stated as a
 * sample: the idle asset carries a fraction of the atlas so the brain has
 * structure before a prompt, and it is not the set a trace is read against.
 */
function BrainLegend({ atlas }: { atlas: Atlas | null | undefined }) {
  if (atlas === undefined) {
    return (
      <div className="pointer-events-none absolute top-3 left-3 max-w-[16rem] rounded-[12px] bg-[#0D0E11]/80 p-2.5 text-[11px]">
        <p className="text-text-tertiary leading-4">Loading the feature atlas…</p>
      </div>
    )
  }

  // 4.2 — the shell alone, and the reason for it.
  if (atlas === null) {
    return (
      <div className="pointer-events-none absolute top-3 left-3 max-w-[16rem] space-y-1 rounded-[12px] bg-[#0D0E11]/80 p-2.5 text-[11px]">
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
    <div className="pointer-events-none absolute top-3 left-3 max-w-[16rem] space-y-2 rounded-[12px] bg-[#0D0E11]/80 p-2.5 text-[11px]">
      <div className="space-y-1">
        <p className="text-text-tertiary font-medium tracking-[0.04em] uppercase">Feature nodes</p>
        <p className="text-text-secondary leading-4">
          A node is one SAE feature — a single direction the sparse autoencoder reads out of the
          residual stream.
        </p>
        <p className="text-text-secondary leading-4">
          {atlas.nodes.length.toLocaleString()} of {atlas.total.toLocaleString()} features,
          sampled — one dot each, across layers {layers.join(', ')}.
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
      </div>

      <div className="space-y-1">
        <p className="text-text-tertiary font-medium tracking-[0.04em] uppercase">Areas</p>
        <p className="text-text-secondary leading-4">
          An area is a cluster of features whose residual-stream directions point close together —
          a group of similar features, nothing more.
        </p>
        <p className="text-text-tertiary leading-4">
          {atlas.areas.length} of them, {named === 0 ? 'none named' : `${named} named`} — an area
          earns a name only when its features&apos; own labels agree more than a random group of
          the same size.
          {atlas.explainerAmi !== null && atlas.source === 'labels' ? (
            <>
              {' '}
              Explainer influence{' '}
              <span className="text-text-primary font-mono tabular-nums">
                {atlas.explainerAmi.toFixed(3)}
              </span>
              : the areas are not an artifact of which model wrote the labels.
            </>
          ) : null}
        </p>
      </div>

      {/* 5.4 — the layout's limits, and the measurement behind them. A picture
          nobody measured is decoration, so the number is on screen rather than
          in a log. */}
      {atlas.knnPreservation !== null ? (
        <p className="text-text-tertiary leading-4">
          Positions come from a projection that preserves what is nearby and distorts what is far:
          of a feature&apos;s {atlas.knnK ?? 20} nearest neighbours,{' '}
          <span className="text-text-primary font-mono tabular-nums">
            {(atlas.knnPreservation * 100).toFixed(0)}%
          </span>{' '}
          survive the flattening to three dimensions. Distance beyond a neighbourhood is
          meaningless, which is why there is no axis, no coordinate and no scale to read.
        </p>
      ) : (
        <p className="text-text-tertiary leading-4">
          Positions come from a projection that preserves what is nearby and distorts what is far.
          This atlas records no fidelity measurement, so how much of the original structure
          survived is unknown — and distance is not readable either way, which is why there is no
          axis, no coordinate and no scale.
        </p>
      )}

      <p className="text-text-tertiary leading-4">
        Neither a node nor an area is a brain region. No named anatomical area computes any of
        this; the shell is a container, not a map.
      </p>
      <p className="text-text-disabled font-mono text-[10px] leading-4">
        atlas {atlas.version} · {atlas.source}
      </p>
    </div>
  )
}

/** What each scope lights, said in words beside the control that sets it. */
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
  areas,
  atlas,
  featureLayers,
  filter,
  lit,
  onQueryChange,
  onScopeChange,
  onSelectLayer,
  query,
  scope,
  selection,
  trace,
}: {
  agreement: ReturnType<typeof atlasAgreement>
  areas: LitArea[]
  atlas: Atlas | null | undefined
  featureLayers: number[]
  filter: LabelFilter
  lit: LitSet
  onQueryChange: (query: string) => void
  onScopeChange: (scope: LitScope) => void
  onSelectLayer?: (layer: number) => void
  query: string
  scope: LitScope
  selection: { layer: number; position: number } | null
  trace: Trace | null
}) {
  if (trace === null) return null

  const coverage = coverageNote(lit, atlas)
  const filtering = filter.query !== ''
  const matched = filter.nodes.length

  return (
    <div className="pointer-events-none absolute top-3 right-3 max-w-[17rem] space-y-2 rounded-[12px] bg-[#0D0E11]/80 p-2.5 text-[11px]">
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

      <Transport
        featureLayers={featureLayers}
        onSelectLayer={onSelectLayer}
        scope={scope}
        selection={selection}
        trace={trace}
      />

      <LabelSearch
        filter={filter}
        onQueryChange={onQueryChange}
        query={query}
      />

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
            <dd className="text-text-primary font-mono tabular-nums">
              {filtering ? `${matched} / ${lit.nodes.length}` : lit.nodes.length}
            </dd>
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

      <AreaReadout areas={areas} />

      {/* 4.4 — a feature that fired but has nowhere honest to be drawn. */}
      {lit.unplaced.length > 0 ? (
        <div className="space-y-1">
          <p className="text-const leading-4">
            {lit.unplaced.length} feature{lit.unplaced.length === 1 ? '' : 's'} the atlas cannot
            place {lit.unplaced.length === 1 ? 'is' : 'are'} not drawn — no position is invented
            for {lit.unplaced.length === 1 ? 'it' : 'them'}.
          </p>
          <ul className="border-border-subtle bg-bg-elevated rounded-[10px] border">
            {lit.unplaced.slice(0, 8).map((feature) => (
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
 * The areas this scope lit, and the two numbers that keep an area's glow
 * honest: how its members were combined, and how many of them fired.
 *
 * Without the second number a bright area reads as "all of this lit up", when
 * it is often one member of four hundred. The count is the difference between
 * a finding and an illusion, so it sits beside every row rather than in a
 * tooltip.
 */
function AreaReadout({ areas }: { areas: LitArea[] }) {
  if (areas.length === 0) return null

  return (
    <div className="space-y-1">
      <p className="text-text-tertiary leading-4">
        {areas.length} area{areas.length === 1 ? '' : 's'} lit — brightness is the{' '}
        <span className="text-text-primary font-mono">{AREA_AGGREGATION}</span> of each area&apos;s
        active members, and how many were active is beside it.
      </p>
      <ul className="border-border-subtle bg-bg-elevated space-y-0.5 rounded-[10px] border px-2 py-1">
        {areas.slice(0, 8).map((entry) => (
          <li className="flex items-baseline justify-between gap-2" key={entry.area.cluster}>
            <span
              className={
                entry.area.name === null
                  ? 'text-text-disabled truncate italic'
                  : 'text-text-secondary truncate'
              }
            >
              {entry.area.name ?? 'unnamed area'}
            </span>
            <span className="text-text-tertiary shrink-0 font-mono tabular-nums">
              {entry.active}/{entry.area.n_members}
            </span>
          </li>
        ))}
      </ul>
    </div>
  )
}

/**
 * The label filter, and everything it has to admit to.
 *
 * Three states, all different and all stated: no query (everything lit is
 * drawn), a query with matches (how many of how many), and a query with none
 * — which must never look like an empty scope. Unlabelled features are
 * reported separately because no query can ever match them: there is no text
 * to match against, and letting them vanish silently would read as the filter
 * having ruled them out.
 */
function LabelSearch({
  filter,
  onQueryChange,
  query,
}: {
  filter: LabelFilter
  onQueryChange: (query: string) => void
  query: string
}) {
  const active = filter.labelled + filter.unlabelled
  if (active === 0) return null

  const filtering = filter.query !== ''
  const nothingToSearch = filter.labelled === 0

  return (
    <div className="space-y-1">
      <input
        aria-label="Filter lit features by label text"
        className="border-border-subtle bg-bg-elevated text-text-primary placeholder:text-text-disabled focus:border-fn/40 pointer-events-auto w-full rounded-[10px] border px-2 py-1 font-mono text-[11px] outline-none"
        disabled={nothingToSearch}
        onChange={(event) => onQueryChange(event.target.value)}
        placeholder={nothingToSearch ? 'no labels to search' : 'filter by label…'}
        type="search"
        value={query}
      />

      {nothingToSearch ? (
        <p className="text-const leading-4">
          None of the active features carry an explanation, so there is nothing to match on.
        </p>
      ) : filtering ? (
        <p
          className={filter.nodes.length === 0 ? 'text-const leading-4' : 'text-text-tertiary leading-4'}
        >
          {filter.nodes.length === 0 ? (
            <>No active feature&apos;s label matched “{filter.query}”.</>
          ) : (
            <>
              <span className="text-text-primary font-mono tabular-nums">
                {filter.nodes.length}
              </span>{' '}
              of {filter.labelled} labelled active features matched.
            </>
          )}
        </p>
      ) : null}

      {filtering && filter.unlabelled > 0 ? (
        <p className="text-text-tertiary leading-4">
          {filter.unlabelled} active feature{filter.unlabelled === 1 ? '' : 's'} carr
          {filter.unlabelled === 1 ? 'ies' : 'y'} no label and cannot be searched — they are
          excluded while a filter is on.
        </p>
      ) : null}
    </div>
  )
}

/**
 * Stepping the selected layer, which is how depth is read now that it is not
 * a spatial axis.
 *
 * It stops at both ends rather than wrapping: L25 followed by L0 would read
 * as a cycle, and the residual stream does not loop. Using it narrows the
 * brain to cell scope, because that is the only scope whose lit set depends
 * on the layer at all — a transport that changed nothing visible would be a
 * control that lies about what it does.
 */
function Transport({
  featureLayers,
  onSelectLayer,
  scope,
  selection,
  trace,
}: {
  featureLayers: number[]
  onSelectLayer?: (layer: number) => void
  scope: LitScope
  selection: { layer: number; position: number } | null
  trace: Trace
}) {
  if (onSelectLayer === undefined || selection === null) return null

  const layer = selection.layer
  const last = trace.n_layers - 1
  // 6.6 — inert, and saying why, on a trace the SAE pass never ran for.
  const inert = featureLayers.length === 0
  const canGoBack = !inert && layer > 0
  const canGoOn = !inert && layer < last
  const hasData = featureLayers.includes(layer)

  return (
    <div className="space-y-1">
      <div className="border-border-subtle bg-bg-elevated pointer-events-auto flex items-center rounded-[10px] border">
        <button
          aria-label="Previous layer"
          className="text-text-tertiary enabled:hover:bg-white/[0.02] disabled:text-text-disabled px-2 py-1 font-mono transition-colors duration-150 disabled:cursor-not-allowed"
          disabled={!canGoBack}
          onClick={() => onSelectLayer(layer - 1)}
          type="button"
        >
          ◀
        </button>
        <span className="text-text-primary flex-1 text-center font-mono tabular-nums">
          L{layer}
          <span className="text-text-tertiary"> of {last}</span>
        </span>
        <button
          aria-label="Next layer"
          className="text-text-tertiary enabled:hover:bg-white/[0.02] disabled:text-text-disabled px-2 py-1 font-mono transition-colors duration-150 disabled:cursor-not-allowed"
          disabled={!canGoOn}
          onClick={() => onSelectLayer(layer + 1)}
          type="button"
        >
          ▶
        </button>
      </div>

      {inert ? (
        <p className="text-const leading-4">
          The transport is inert: this trace records no features, so stepping depth would change
          nothing on the brain.
        </p>
      ) : !hasData ? (
        <p className="text-const leading-4">
          The SAE pass did not run at L{layer}, so nothing is lit from it.
        </p>
      ) : scope !== 'cell' ? (
        <p className="text-text-tertiary leading-4">
          Stepping narrows to cell scope — the only scope whose lit set depends on the layer.
        </p>
      ) : null}
    </div>
  )
}

/**
 * One area, under the pointer — and where its name came from.
 *
 * A name here is one member feature's own label standing for hundreds of
 * others. That is the same lossiness a blended band had, and it is disclosed
 * the same way: the member is named, so a reader can go and check whether the
 * sentence describes the neighbourhood or just the one feature it was taken
 * from. The measurement that let it be named at all — its members' label
 * agreement against a shuffled baseline of the same size — is shown beside
 * it, because "earned a name" is a threshold result and the threshold is the
 * interesting part.
 *
 * An unnamed area shows its measurement too. It is not a gap in the data: it
 * is the atlas having looked and found no agreement worth naming, which is a
 * finding about the cluster and is reported as one.
 */
function AreaDetail({ entry }: { entry: { area: AtlasArea; lit: LitArea | null } }) {
  const { area, lit } = entry
  const source = area.name_source ?? null

  return (
    <div className="border-border-subtle bg-bg-elevated absolute bottom-3 left-3 max-w-[20rem] space-y-1 rounded-[10px] border px-2.5 py-2 text-[11px]">
      <p className={area.name === null ? 'text-text-disabled italic' : 'text-text-secondary'}>
        {area.name ?? 'unnamed area'}
      </p>

      <p className="text-text-tertiary font-mono tabular-nums">
        area {area.cluster} · {area.n_members.toLocaleString()} features
        {lit === null ? null : (
          <>
            {' '}
            · <span className="text-text-primary">{lit.active} active</span> · peak{' '}
            {lit.activation.toFixed(2)}
          </>
        )}
      </p>

      {/* Whose label this is. Without it the name reads as a summary of the
          cluster, which nothing here measured it to be. */}
      {area.name !== null ? (
        source === null ? (
          <p className="text-text-tertiary leading-4">
            This atlas did not record which member the name came from.
          </p>
        ) : (
          <p className="text-text-tertiary leading-4">
            Named from{' '}
            <span className="text-text-primary font-mono">
              L{source[0]} #{source[1]}
            </span>
            &apos;s own label — the cluster&apos;s medoid, one member speaking for{' '}
            {area.n_members.toLocaleString()}.
          </p>
        )
      ) : null}

      {area.coherence === null ? (
        <p className="text-text-tertiary leading-4">
          No naming was attempted: the label store held no embeddings to measure agreement with.
        </p>
      ) : (
        <p className="text-text-tertiary leading-4">
          Label agreement{' '}
          <span className="text-text-primary font-mono tabular-nums">
            {area.coherence.toFixed(3)}
          </span>
          {area.baseline_coherence === null ? null : (
            <>
              {' '}
              against{' '}
              <span className="font-mono tabular-nums">
                {area.baseline_coherence.toFixed(3)}
              </span>{' '}
              for a random group of the same size
            </>
          )}
          {area.name === null ? ' — not enough of a margin to earn a name.' : '.'}
        </p>
      )}

      {area.name_withheld ? (
        <p className="text-const leading-4">{area.name_withheld}</p>
      ) : null}

      {area.explainers ? (
        <p className="text-text-disabled font-mono text-[10px] leading-4">
          labelled by {area.explainers}
        </p>
      ) : null}
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

// --------------------------------------------------------------------------
// activity rendering
// --------------------------------------------------------------------------

/** How far the shell lifts while a job is working. */
const WORKING_LIFT = 0.14

/**
 * The shell's one concession to a job in flight: a flat, undifferentiated
 * lift for as long as something is running.
 *
 * It names no layer and no position. The old sweep walked a band per reported
 * layer, and there are no bands left to walk — depth is not a spatial axis
 * here, so there is nowhere on the shell a layer could be. Showing which
 * layers have arrived is the nodes' job, on the features themselves.
 */
function applyActivity(handles: SceneHandles | null, activity: Activity): void {
  if (!handles) return
  const lift = activity.kind === 'idle' ? 0 : WORKING_LIFT
  // One write per state change, not one per frame.
  if (lift === handles.lift) return
  handles.lift = lift
  handles.material.color.setRGB(
    Math.min(1, SHELL_COLOR.r + lift),
    Math.min(1, SHELL_COLOR.g + lift),
    Math.min(1, SHELL_COLOR.b + lift),
  )
}

/** Hermite fade between two thresholds. */
function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = Math.min(Math.max((x - edge0) / (edge1 - edge0), 0), 1)
  return t * t * (3 - 2 * t)
}

/**
 * The whole of the progressive disclosure, once per frame.
 *
 * Distance to the camera is the only input: far out, the area masses carry
 * the picture and their names are readable, so the overview needs no single
 * node's identity; close in, the masses thin out and the nodes grow into
 * things worth aiming at. Nothing here is a React render — the brain turns at
 * 60fps and re-rendering the component to move a label would cost more than
 * everything else in this file put together.
 */
function applyDetail(handles: SceneHandles | null): void {
  if (!handles) return

  const { camera, brainGroup } = handles
  const distance = camera.position.length()
  const detail = detailLevel(distance, handles.minDistance, handles.maxDistance)
  handles.detail = detail

  const active = handles.activeCloud?.material as THREE.ShaderMaterial | undefined
  if (active) active.uniforms.detail.value = detail
  const areas = handles.areaCloud?.material as THREE.ShaderMaterial | undefined
  if (areas) areas.uniforms.detail.value = detail
  const idle = handles.nodeCloud?.material as THREE.PointsMaterial | undefined
  // Scaled around the default framing (~0.62), so the idle cloud looks the
  // way it always has until the camera actually moves.
  if (idle) idle.size = NODE_SIZE * (0.72 + detail * 0.45)

  if (handles.areaLabels.length === 0) return

  const fade = 1 - smoothstep(LABEL_FADE_START, LABEL_FADE_END, detail)
  if (fade <= 0.01) {
    for (const { element } of handles.areaLabels) element.style.visibility = 'hidden'
    return
  }

  const canvas = handles.renderer.domElement
  const width = canvas.clientWidth
  const height = canvas.clientHeight

  brainGroup.updateMatrixWorld()
  const world = new THREE.Vector3()
  const candidates: { entry: SceneHandles['areaLabels'][number]; x: number; y: number; depth: number }[] = []

  for (const entry of handles.areaLabels) {
    world.copy(entry.centroid).applyMatrix4(brainGroup.matrixWorld)
    const depth = camera.position.distanceTo(world)
    world.project(camera)
    if (world.z > 1) {
      // Behind the camera entirely.
      entry.element.style.visibility = 'hidden'
      continue
    }
    candidates.push({
      entry,
      x: (world.x * 0.5 + 0.5) * width,
      y: (-world.y * 0.5 + 0.5) * height,
      depth,
    })
  }

  // Who gets to speak when two names land on the same pixels: what the trace
  // lit first, then whichever is nearer the camera. Without this the atlas's
  // centroids — which really are close together, that being what the layout
  // measured — stack into a wall of text that says nothing at all. A hidden
  // name is not lost: rotate or zoom and it takes its turn.
  candidates.sort((a, b) =>
    a.entry.lit === b.entry.lit ? a.depth - b.depth : a.entry.lit ? -1 : 1,
  )

  const placed: { x: number; y: number; w: number; h: number }[] = []
  for (const { entry, x, y, depth } of candidates) {
    const w = entry.width + LABEL_GAP
    const h = entry.height + LABEL_GAP
    const clash = placed.some(
      (box) =>
        Math.abs(box.x - x) * 2 < box.w + w && Math.abs(box.y - y) * 2 < box.h + h,
    )
    if (clash) {
      entry.element.style.visibility = 'hidden'
      continue
    }
    placed.push({ x, y, w, h })

    entry.element.style.visibility = 'visible'
    entry.element.style.transform = `translate3d(${x.toFixed(1)}px, ${y.toFixed(1)}px, 0) translate(-50%, -50%)`
    // A label on the far side of the shell dims, so the near ones read first.
    const behind = smoothstep(distance - 0.6, distance + 1.0, depth)
    entry.element.style.opacity = (fade * (1 - behind * 0.55)).toFixed(3)
  }
}

/**
 * What a reported layer means for the picture, while the job is still running.
 *
 * The service reports how far a layer-counting phase has got, but a trace's
 * features only reach the browser with the finished trace — there is no
 * partial-feature channel. So the honest reading of "the pass has reached
 * layer 14" is that layer 14 is *being computed*, not that anything from it
 * can be lit: no activation for it has been received, and lighting nodes on
 * the strength of a counter would be inventing the very data the counter is
 * counting towards.
 */
function ComputingNote({ activity }: { activity: Activity }) {
  if (activity.kind !== 'layers' || activity.total === 0) return null

  return (
    <div className="border-border-subtle bg-[#0D0E11]/85 text-text-tertiary pointer-events-none absolute top-12 right-3 max-w-[15rem] rounded-[10px] border px-2.5 py-2 text-[11px] leading-4">
      <p>
        Computing layer{' '}
        <span className="text-text-primary font-mono tabular-nums">{activity.done}</span> of{' '}
        {activity.total}. Its features arrive with the finished trace, so nothing is lit from it
        yet — the brain never runs ahead of what the service has sent.
      </p>
    </div>
  )
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
      : activity.kind === 'layers'
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

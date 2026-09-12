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
import type { LitArea, LitNode, LitScope, LitSet } from '@/lib/lit'
import {
  AGGREGATION,
  AREA_AGGREGATION,
  BOS_REASON,
  atlasAgreement,
  coverageNote,
  litAreas,
  litSet,
  prominence,
} from '@/lib/lit'
import { KdTree, rayPoints } from '@/lib/kdtree'
import type { RunState } from '@/hooks/useTrace'

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
 * lens pass — and it reports its position in that count without claiming to
 * show where it is: depth is not a spatial axis here, so there is nowhere in
 * the cloud for a layer to be.
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
  /**
   * True while something else holds the stage — the grid view, today.
   *
   * The scene stays mounted rather than unmounting, because the camera is a
   * held state: a reader orbits to a cluster, checks the grid, and coming back
   * to a reset camera would make the toggle cost something. Mounted and hidden
   * is only cheap if it stops drawing, hence this — the loop keeps its frame
   * callback alive so it can resume, and renders nothing meanwhile.
   */
  paused?: boolean
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
 * background, and `depthWrite: false` so a nearer translucent point does not
 * punch a hole in the one behind it.
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
  // The cloud is rotated by its group, so it must never be culled by a
  // bounding sphere computed before that rotation is applied.
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
/** The weakest a lit node may be drawn. See the active fragment shader. */
const ACTIVE_MIN_ALPHA = 0.7

/**
 * How far the idle cloud drops back once something is lit.
 *
 * The idle sample is context — "features exist here" — and the lit set is the
 * finding. At cell scope the lit set is ~16 nodes against 20,000, so without
 * this the data is a rounding error on the texture behind it.
 */
const NODE_IDLE_OPACITY_LIT = 0.07

// --------------------------------------------------------------------------
// the trail: the layers the sweep has already passed
// --------------------------------------------------------------------------

/**
 * Why a trail exists at all.
 *
 * A cell-scope frame answers "what fired at L14" and says nothing whatever
 * about order — and order is the one thing a residual stream actually has.
 * Stepping the transport by hand shows 26 unrelated frames and leaves the
 * sequence to be held in the reader's head, which is the same as not showing
 * it. So the layers already passed stay on screen behind the current one,
 * fading with age, and depth becomes something you watch rather than
 * something you reconstruct.
 *
 * A separate colour, not a dimmer green: brightness alone reads as "weaker
 * activation", which is a claim about the model, and this is a claim about
 * time. Blue is *earlier*, green is *now*, and nothing is being said about
 * strength by the difference.
 */
const TRAIL_COLOR = new THREE.Color(0x82aaff)

/** How many passed layers stay on screen behind the current one. */
const TRAIL_DEPTH = 6

/** Milliseconds per layer while the sweep runs. */
const SWEEP_MS = 280

const TRAIL_VERTEX_SHADER = `
attribute float prominence;
attribute float decay;
uniform float detail;
varying float vFade;
void main() {
  vFade = prominence * decay;
  vec4 viewPosition = modelViewMatrix * vec4(position, 1.0);
  // Older layers shrink as well as dim, so the head of the sweep stays the
  // thing being looked at even where the trail is dense.
  gl_PointSize = mix(${ACTIVE_MIN_SIZE.toFixed(3)}, ${ACTIVE_MAX_SIZE.toFixed(3)}, prominence)
    * mix(0.55, 1.0, decay)
    * mix(0.85, 1.45, detail)
    * (300.0 / -viewPosition.z);
  gl_Position = projectionMatrix * viewPosition;
}
`

const TRAIL_FRAGMENT_SHADER = `
uniform vec3 color;
varying float vFade;
void main() {
  vec2 offset = gl_PointCoord - vec2(0.5);
  float radius = length(offset);
  if (radius > 0.5) discard;
  float edge = smoothstep(0.5, 0.15, radius);
  gl_FragColor = vec4(color, edge * vFade * 0.5);
}
`

/** One node from a layer the sweep has passed, with how far back it was. */
interface TrailNode extends LitNode {
  /** 1 is the layer just left; TRAIL_DEPTH is the oldest still drawn. */
  age: number
}

/**
 * Normalised against the trail's own peak, not the current layer's.
 *
 * Same rule as `prominence` documents — brightest node in the same set — and
 * it has to be applied to this set separately. The head of the sweep is one
 * layer and the trail is six, so sharing a denominator would make the trail
 * behind a cold layer clamp to full brightness and the trail behind a hot one
 * vanish, for a reason about which layer happens to be current rather than
 * about what fired.
 */
function buildTrailCloud(nodes: TrailNode[]): THREE.Points {
  let maxActivation = 0
  for (const node of nodes) if (node.activation > maxActivation) maxActivation = node.activation

  const positions = new Float32Array(nodes.length * 3)
  const prominences = new Float32Array(nodes.length)
  const decays = new Float32Array(nodes.length)
  for (let i = 0; i < nodes.length; i++) {
    positions[i * 3] = nodes[i].x
    positions[i * 3 + 1] = nodes[i].y
    positions[i * 3 + 2] = nodes[i].z
    prominences[i] = prominence(nodes[i].activation, maxActivation)
    decays[i] = 1 - (nodes[i].age - 1) / TRAIL_DEPTH
  }

  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3))
  geometry.setAttribute('prominence', new THREE.BufferAttribute(prominences, 1))
  geometry.setAttribute('decay', new THREE.BufferAttribute(decays, 1))

  const material = new THREE.ShaderMaterial({
    uniforms: { color: { value: TRAIL_COLOR }, detail: { value: 0 } },
    vertexShader: TRAIL_VERTEX_SHADER,
    fragmentShader: TRAIL_FRAGMENT_SHADER,
    transparent: true,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  })

  const points = new THREE.Points(geometry, material)
  points.frustumCulled = false
  return points
}

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

/** Slightly outside the cloud, so a node right at its edge is reachable. */
const PICK_SPHERE_RADIUS = 1.6
const PICK_STEPS = 96
/** How near the ray a node must pass to count as hovered, in scene units. */
const PICK_RADIUS = 0.055
/**
 * How far the pointer may travel between press and release and still count as
 * a click rather than a drag. The same gesture orbits the brain, so without
 * this every turn of the camera would re-pin whatever ended up under the
 * cursor.
 */
const CLICK_SLOP = 4

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
  // The alpha floor is the weakest a lit node is allowed to be. It used to be
  // 0.35, which put a low-activation node under the idle cloud's own
  // brightness: green became grey-green and the lit set stopped reading as
  // the data. A lit node is a finding at any activation, so the floor is high
  // enough to be unambiguously green, and prominence still separates the
  // strong from the weak across the range above it.
  gl_FragColor = vec4(color, edge * mix(${ACTIVE_MIN_ALPHA.toFixed(2)}, 1.0, vProminence));
}
`

/**
 * The ring around a pinned node.
 *
 * A pin holds one node's panel open, and a held-open panel with nothing on
 * screen to say *which* node it belongs to is a caption without a subject —
 * the reader has to remember what they clicked. So the pin is drawn.
 *
 * A ring rather than a filled dot, and in the interface's own tone rather than a
 * brighter green: the node inside it has to stay visible and stay the colour
 * its activation earned. The marker says "this one is being read", which is a
 * fact about the reader, not about the model.
 */
const PIN_COLOR = new THREE.Color(0xe6e8eb)
/** Comfortably clear of ACTIVE_MAX_SIZE, so the ring surrounds the node. */
const PIN_SIZE = 0.17

const PIN_VERTEX_SHADER = `
uniform float detail;
void main() {
  vec4 viewPosition = modelViewMatrix * vec4(position, 1.0);
  gl_PointSize = ${PIN_SIZE.toFixed(3)} * mix(0.85, 1.45, detail) * (300.0 / -viewPosition.z);
  gl_Position = projectionMatrix * viewPosition;
}
`

const PIN_FRAGMENT_SHADER = `
uniform vec3 color;
void main() {
  float radius = length(gl_PointCoord - vec2(0.5));
  if (radius > 0.5) discard;
  // Two smoothsteps make an annulus: opaque between them, transparent inside
  // and out, so the node keeps reading through the hole in the middle.
  float ring = smoothstep(0.33, 0.40, radius) * smoothstep(0.50, 0.43, radius);
  gl_FragColor = vec4(color, ring * 0.85);
}
`

function buildPinMarker(node: LitNode): THREE.Points {
  const geometry = new THREE.BufferGeometry()
  geometry.setAttribute(
    'position',
    new THREE.BufferAttribute(new Float32Array([node.x, node.y, node.z]), 3),
  )

  const material = new THREE.ShaderMaterial({
    uniforms: { color: { value: PIN_COLOR }, detail: { value: 0 } },
    vertexShader: PIN_VERTEX_SHADER,
    fragmentShader: PIN_FRAGMENT_SHADER,
    transparent: true,
    depthWrite: false,
    // Not additive: the ring is a marker, not light, and adding it to a dense
    // patch of cloud would blow out to white exactly where it needs to read
    // as an outline.
    blending: THREE.NormalBlending,
  })

  const points = new THREE.Points(geometry, material)
  points.frustumCulled = false
  return points
}

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
  /** The atlas's node cloud, once an atlas has loaded. Null until then. */
  nodeCloud: THREE.Points | null
  /** The features the current trace lit, at the current scope. */
  activeCloud: THREE.Points | null
  /** The layers the sweep has passed, fading with age. Null when not sweeping. */
  trailCloud: THREE.Points | null
  /** The lit areas, as soft masses at their centroids. */
  areaCloud: THREE.Points | null
  /** The ring around the pinned node, if one is pinned. */
  pinMarker: THREE.Points | null
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
  /** 0 (furthest) to 1 (closest), recomputed each frame from the camera. */
  detail: number
  /** Orbit limits, for turning camera distance into `detail`. */
  minDistance: number
  maxDistance: number
}

export function Brain({
  trace,
  selection,
  status,
  progress,
  onSelectLayer,
  paused = false,
}: BrainProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const handlesRef = useRef<SceneHandles | null>(null)

  // Read by the animation loop, which is set up once and must not tear down
  // and rebuild the scene every time the stage switches views. Written from an
  // effect rather than during render: the loop is an external system, and the
  // one frame it may still draw before the flag lands is drawn into a
  // container that is already hidden.
  const pausedRef = useRef(paused)
  useEffect(() => {
    pausedRef.current = paused
  }, [paused])

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

  // Which node has been clicked, if any.
  //
  // Hover alone made a node something you could look at and never touch: the
  // panel carries a Neuronpedia link, and moving the pointer towards that link
  // takes it off the node, which closes the panel the link was in. A pin is
  // the fix — the panel stays until it is dismissed, so the link is reachable
  // and the text is selectable.
  //
  // It outranks hover rather than yielding to it. A pinned panel that swapped
  // its contents for whatever the pointer crossed on the way to the link would
  // be the same trap with an extra step.
  const [pinnedNode, setPinnedNode] = useState<LitNode | null>(null)

  // Which area's label the pointer is over. Carries the atlas's record and
  // what this trace lit inside it, because the disclosure needs both.
  const [hoveredArea, setHoveredArea] = useState<{
    area: AtlasArea
    lit: LitArea | null
  } | null>(null)


  const labelsRef = useRef<HTMLDivElement>(null)

  // The features this trace lit. Positions come from `trace.layout`, never
  // from the atlas asset — see litSet's own note on why that distinction is
  // not cosmetic.
  const lit = useMemo<LitSet>(() => litSet(trace, scope, selection), [trace, scope, selection])

  const drawn = lit.nodes

  // A new lit set invalidates the pin: it names one node in the set being
  // replaced, and holding it open would leave a panel — and a ring in the
  // scene — describing a feature that is no longer drawn. Adjusted during
  // render rather than in an effect, the same shape as `lastTrace` below, so
  // the stale panel is never committed and painted first.
  const [pinnedIn, setPinnedIn] = useState(drawn)
  if (drawn !== pinnedIn) {
    setPinnedIn(drawn)
    setPinnedNode(null)
  }

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

  // -- the sweep: depth as motion -----------------------------------------
  //
  // The transport walks one layer per click, which shows 26 frames and leaves
  // the order between them to be remembered. Playing it walks depth on a
  // clock instead, and `trail` keeps the layers already passed on screen
  // behind the current one — so "what fired in what order" is watched rather
  // than reconstructed. See `TRAIL_COLOR` for why the trail is a different
  // colour and not a dimmer one.
  const [playing, setPlaying] = useState(false)

  // Every layer the sweep has passed and not yet aged out, oldest first.
  const [trail, setTrail] = useState<{ layer: number; nodes: LitNode[] }[]>([])

  const sweepLayer = selection?.layer ?? null
  const lastLayer = trace === null ? null : trace.n_layers - 1

  // A new trace stops the sweep and drops the history with it: that history
  // belongs to a trace that is no longer on screen. Adjusted during render
  // rather than in an effect, for the same reason `lastMoved` below is —
  // an effect would commit one painted frame of the old sweep first.
  const [lastTrace, setLastTrace] = useState(trace)
  if (trace !== lastTrace) {
    setLastTrace(trace)
    setPlaying(false)
    setTrail([])
  }

  // The history itself, accumulated at each layer the sweep lands on.
  //
  // It cannot be derived during render from the current props, because it is
  // what the props *were* — so it is written against the layer last recorded,
  // the same shape as the selection adjustment below. Keyed by layer so a
  // sweep that revisits one replaces its entry rather than drawing it twice,
  // and capped at `TRAIL_DEPTH` so the oldest layers leave instead of piling
  // into a static haze that says nothing about order.
  const [trailAt, setTrailAt] = useState<number | null>(null)
  if (playing && sweepLayer !== null && sweepLayer !== trailAt) {
    setTrailAt(sweepLayer)
    setTrail((previous) => [
      ...previous.filter((entry) => entry.layer !== sweepLayer).slice(-TRAIL_DEPTH),
      { layer: sweepLayer, nodes: drawn },
    ])
  }

  // One layer per tick, stopping at the end rather than wrapping — L25
  // followed by L0 would read as a cycle, and the residual stream does not
  // loop. Same reason the transport's arrows stop at both ends. The last
  // layer holds for a full beat before the sweep stops, so the end of the
  // run is something you see rather than something that blinks past.
  useEffect(() => {
    if (!playing) return
    if (onSelectLayer === undefined || sweepLayer === null || lastLayer === null) return
    const timer = window.setTimeout(() => {
      if (sweepLayer >= lastLayer) setPlaying(false)
      else onSelectLayer(sweepLayer + 1)
    }, SWEEP_MS)
    return () => window.clearTimeout(timer)
  }, [playing, onSelectLayer, sweepLayer, lastLayer])

  // Flattened for drawing, with the current layer left out: it is already the
  // active cloud, and drawing it twice under additive blending would make the
  // head of the sweep brighter for a reason about compositing rather than
  // about activation.
  const trailNodes = useMemo<TrailNode[]>(() => {
    if (!playing) return []
    const past = trail.filter((entry) => entry.layer !== sweepLayer)
    const out: TrailNode[] = []
    past.forEach((entry, index) => {
      const age = past.length - index
      for (const node of entry.nodes) out.push({ ...node, age })
    })
    return out
  }, [playing, trail, sweepLayer])

  // The activity the brain is entitled to show, derived from the job's own
  // reported state. It counts, and never places: a layer counter is a number,
  // not a position in the cloud.
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
    // Close enough to resolve one node, far enough to see the whole cloud.
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

    // RenderPass and OutputPass only: OutputPass is what applies the tone
    // mapping and sRGB conversion, so it earns its place. There is no bloom —
    // a halo around the cloud is decoration, and nothing in this interface
    // glows.
    const composer = new EffectComposer(renderer)
    composer.addPass(new RenderPass(scene, camera))
    composer.addPass(new OutputPass())

    handlesRef.current = {
      renderer,
      composer,
      camera,
      brainGroup,
      nodeCloud: null,
      activeCloud: null,
      trailCloud: null,
      areaCloud: null,
      pinMarker: null,
      areaLabels: [],
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
    // Accumulated from per-frame deltas rather than read off the clock, so a
    // pause holds the turn where it was instead of letting wall-clock time run
    // on underneath and snapping the brain round on the way back.
    let spin = 0
    const animate = () => {
      frameId = requestAnimationFrame(animate)

      const delta = clock.getDelta()
      if (pausedRef.current) return
      spin += delta

      if (!userDriven) {
        brainGroup.rotation.y = stillness.matches ? 0.35 : spin * 0.04
      }

      applyDetail(handlesRef.current)

      controls.update()
      composer.render()
    }
    animate()

    return () => {
      cancelAnimationFrame(frameId)
      resizeObserver.disconnect()
      controls.dispose()
      container.removeChild(renderer.domElement)
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

  // -- the trail: the passed layers, rebuilt as the sweep advances --------
  useEffect(() => {
    const handles = handlesRef.current
    if (!handles || trailNodes.length === 0 || !positionsUsable) return

    const cloud = buildTrailCloud(trailNodes)
    handles.brainGroup.add(cloud)
    handles.trailCloud = cloud

    return () => {
      handles.brainGroup.remove(cloud)
      cloud.geometry.dispose()
      ;(cloud.material as THREE.Material).dispose()
      handles.trailCloud = null
    }
  }, [trailNodes, positionsUsable])

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
      // A name can land over a bright patch of cloud, so the contrast has to
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

    /** The frontmost node under the pointer, or null. Shared by hover and click. */
    const pick = (event: PointerEvent): LitNode | null => {
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
      const span = intersectSphere(origin, direction, PICK_SPHERE_RADIUS)
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
      return found
    }

    const onMove = (event: PointerEvent) => setHoveredNode(pick(event))
    const onLeave = () => setHoveredNode(null)

    // Click and drag arrive as the same pair of events, and the drag orbits the
    // camera — so they are told apart by how far the pointer travelled.
    let pressedAt: { x: number; y: number } | null = null
    const onDown = (event: PointerEvent) => {
      pressedAt = { x: event.clientX, y: event.clientY }
    }
    const onUp = (event: PointerEvent) => {
      const start = pressedAt
      pressedAt = null
      if (start === null) return
      if (Math.hypot(event.clientX - start.x, event.clientY - start.y) > CLICK_SLOP) return
      // A click on empty space picks nothing and so unpins: the way out is the
      // same gesture as the way in, and needs no target to aim at.
      setPinnedNode(pick(event))
    }

    const onKey = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setPinnedNode(null)
    }

    container.addEventListener('pointermove', onMove)
    container.addEventListener('pointerleave', onLeave)
    container.addEventListener('pointerdown', onDown)
    container.addEventListener('pointerup', onUp)
    window.addEventListener('keydown', onKey)
    return () => {
      container.removeEventListener('pointermove', onMove)
      container.removeEventListener('pointerleave', onLeave)
      container.removeEventListener('pointerdown', onDown)
      container.removeEventListener('pointerup', onUp)
      window.removeEventListener('keydown', onKey)
    }
  }, [tree])

  // -- the pin marker: a ring on the node whose panel is being held open ----
  useEffect(() => {
    const handles = handlesRef.current
    if (!handles || pinnedNode === null) return

    const marker = buildPinMarker(pinnedNode)
    // Seeded from the camera's current distance, so the ring is the right size
    // on its first painted frame rather than on the second.
    ;(marker.material as THREE.ShaderMaterial).uniforms.detail.value = handles.detail
    handles.brainGroup.add(marker)
    handles.pinMarker = marker

    return () => {
      handles.brainGroup.remove(marker)
      marker.geometry.dispose()
      ;(marker.material as THREE.Material).dispose()
      handles.pinMarker = null
    }
  }, [pinnedNode])

  // Pressing play at the last layer restarts from L0. The sweep has one
  // direction, so "again" can only mean from the top; doing nothing would be
  // a control that looks live and is not.
  const togglePlay = () => {
    if (playing) {
      setPlaying(false)
      setTrail([])
      return
    }
    if (onSelectLayer !== undefined && sweepLayer !== null && lastLayer !== null) {
      if (sweepLayer >= lastLayer) onSelectLayer(0)
    }
    // A fresh run starts from nothing: the previous sweep's trail behind a new
    // one would read as layers this run had already passed.
    setTrail([])
    setTrailAt(null)
    setPlaying(true)
  }

  // Stepping by hand takes the sweep over rather than fighting it: a clock
  // that kept advancing under the reader's own clicks would make the arrows
  // feel broken.
  const stepLayer =
    onSelectLayer === undefined
      ? undefined
      : (layer: number) => {
          setPlaying(false)
          setTrail([])
          onSelectLayer(layer)
        }

  return (
    <div className="relative h-full w-full">
      <div className="h-full w-full" ref={containerRef} />

      {/* Area names, over the canvas and moved by the animation loop. */}
      <div className="pointer-events-none absolute inset-0 overflow-hidden" ref={labelsRef} />

      <BrainLegend atlas={atlas} resting={trace === null} />

      {/* One right-hand column, not three things each claiming `top-3 right-3`.
          The chip appears while a run is in flight and the panel appears once
          a trace exists, so re-running with a trace already loaded put both at
          the same anchor — they overlapped, and the count was the thing that
          lost. Stacked in source order and spaced by the container, so
          whichever of them is mounted lands under the one above it. */}
      <div className="pointer-events-none absolute top-3 right-3 flex max-w-[17rem] flex-col items-end gap-2">
        <ActivityChip activity={activity} status={status} />
        <ComputingNote activity={activity} />

        <FeaturePanel
          agreement={agreement}
          areas={areas}
          atlas={atlas}
          featureLayers={featureLayers}
          lit={lit}
          onScopeChange={setScope}
          onSelectLayer={stepLayer}
          onTogglePlay={togglePlay}
          playing={playing}
          scope={scope}
          selection={selection}
          trace={trace}
        />
      </div>

      {/* One detail panel. A pin outranks everything, because it is the one
          state the reader asked for out loud. Below it the area wins over a
          node: the pointer is over the area's label, which sits above the
          cloud, so a node behind it is not what is being asked about. */}
      {pinnedNode ? (
        <NodeDetail node={pinnedNode} onUnpin={() => setPinnedNode(null)} trace={trace} />
      ) : hoveredArea ? (
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
function BrainLegend({
  atlas,
  resting,
}: {
  atlas: Atlas | null | undefined
  /**
   * True before any trace exists — the first thing a reader ever sees here.
   *
   * The counts chip below is the right label for someone who already knows
   * what a node and an area are. For someone who does not, "12,345 nodes · 32
   * areas" is two counts of things they have no concept for, sitting on the
   * largest object on screen. At rest the same control says what the cloud is
   * and what will happen to it instead, and the counts come back once a trace
   * has taught them the words.
   */
  resting: boolean
}) {
  // Collapsed by default. Every clause below is still on the page, one click
  // away — but a permanent wall of prose over the canvas was reading as the
  // subject, and the cloud it disclaims was reading as its background.
  const [open, setOpen] = useState(false)

  if (atlas === undefined) {
    return (
      <div className="border-border-subtle bg-bg-elevated pointer-events-none absolute top-3 left-3 rounded-[2px] border px-2 py-1 text-[11px]">
        <p className="text-text-tertiary leading-4">Loading the feature atlas…</p>
      </div>
    )
  }

  // 4.2 — nothing to draw, and the reason for it.
  if (atlas === null) {
    return (
      <div className="border-border-subtle bg-bg-elevated pointer-events-none absolute top-3 left-3 max-w-[16rem] space-y-1 rounded-[2px] border p-2.5 text-[11px]">
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

  // Before a trace: what the cloud is, and the sentence that connects it to
  // the prompt box in the other column. Prose, so it is sans and selectable —
  // only the affordance is a control.
  if (!open && resting) {
    return (
      <div className="border-border-subtle bg-bg-elevated absolute top-3 left-3 max-w-[19rem] space-y-2 rounded-[2px] border p-2.5">
        <p className="text-text-secondary text-[12px] leading-[1.5]">
          A sample of gemma-2-2b&apos;s features &mdash;{' '}
          <span className="text-text-primary font-mono tabular-nums">
            {atlas.nodes.length.toLocaleString()}
          </span>{' '}
          of{' '}
          <span className="text-text-primary font-mono tabular-nums">
            {atlas.total.toLocaleString()}
          </span>
          , arranged so similar ones sit together. Run a prompt to light the ones that
          fire.
        </p>
        <button
          aria-expanded={false}
          className="text-text-tertiary hover:text-text-primary pointer-events-auto text-[12px] underline decoration-dotted underline-offset-2 transition-colors duration-150"
          onClick={() => setOpen(true)}
          type="button"
        >
          What is this?
        </button>
      </div>
    )
  }

  // With a trace on screen, the compact label: what is drawn, counted, and the
  // way back to why.
  if (!open) {
    return (
      <button
        aria-expanded={false}
        className="border-border-subtle bg-bg-elevated text-text-tertiary hover:text-text-primary hover:border-border-strong absolute top-3 left-3 rounded-[2px] border px-2 py-1 font-mono text-[11px] tabular-nums transition-colors duration-150"
        onClick={() => setOpen(true)}
        type="button"
      >
        {atlas.nodes.length.toLocaleString()} nodes · {atlas.areas.length} areas
        <span className="text-text-disabled"> · what is this?</span>
      </button>
    )
  }

  return (
    <div className="border-border-subtle pointer-events-auto absolute top-3 left-3 max-h-[calc(100%-1.5rem)] max-w-[16rem] space-y-2 overflow-y-auto bg-bg-elevated rounded-[2px] border p-2.5 text-[11px]">
      <div className="space-y-1">
        <div className="flex items-baseline justify-between gap-2">
          <p className="text-text-tertiary font-medium tracking-[0.04em] uppercase">
            Feature nodes
          </p>
          <button
            aria-label="Collapse the legend"
            className="text-text-tertiary hover:text-text-primary pointer-events-auto font-mono transition-colors duration-150"
            onClick={() => setOpen(false)}
            type="button"
          >
            ×
          </button>
        </div>
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
        this; the cloud is a layout, not a map.
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
  lit,
  onScopeChange,
  onSelectLayer,
  onTogglePlay,
  playing,
  scope,
  selection,
  trace,
}: {
  agreement: ReturnType<typeof atlasAgreement>
  areas: LitArea[]
  atlas: Atlas | null | undefined
  featureLayers: number[]
  lit: LitSet
  onScopeChange: (scope: LitScope) => void
  onSelectLayer?: (layer: number) => void
  onTogglePlay: () => void
  playing: boolean
  scope: LitScope
  selection: { layer: number; position: number } | null
  trace: Trace | null
}) {
  if (trace === null) return null

  const coverage = coverageNote(lit, atlas)

  return (
    <div className="border-border-subtle bg-bg-elevated pointer-events-none w-full space-y-2 rounded-[2px] border p-2.5 text-[11px]">
      <p className="text-text-tertiary font-medium tracking-[0.04em] uppercase">Lit features</p>

      {/* 4.6 — the scope, and what it means, stated rather than implied. */}
      <div className="border-border-subtle bg-bg-elevated pointer-events-auto flex rounded-[2px] border">
        {(['cell', 'token', 'trace'] as const).map((option) => (
          <button
            className={`flex-1 px-2 py-1 font-mono transition-colors duration-150 first:rounded-l-[2px] last:rounded-r-[2px] ${
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
        onTogglePlay={onTogglePlay}
        playing={playing}
        scope={scope}
        selection={selection}
        trace={trace}
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
        <dl className="text-text-secondary leading-4">
          {/* 4.8 — the slice never reads as the whole, and it says so on the
              same line as the count it qualifies rather than one below it.
              `drawn` alone reads as "this many fired"; the SAE pass keeps the
              strongest 16 of a mean 78, so that reading is wrong by a factor
              of five. Two rows became one without the caveat moving off
              screen — which is the only kind of simplification this number is
              allowed to get. */}
          <div className="flex justify-between gap-2">
            <dt>drawn</dt>
            <dd className="text-text-primary font-mono tabular-nums">
              {lit.nodes.length}
              {lit.fired === null ? (
                ''
              ) : (
                <span className="text-text-tertiary"> of {lit.fired} fired</span>
              )}
            </dd>
          </div>
        </dl>
      )}

      <SetDetail aggregation={lit.aggregation} areas={areas} />

      <Caveats coverage={coverage} lit={lit} />
    </div>
  )
}

/**
 * Everything true of the lit set that is not a control and not a count.
 *
 * Folded behind one disclosure, not deleted. Each of these exists to stop a
 * specific misreading — a slice read as the whole, an unplaceable feature read
 * as one that never fired, an unrun layer read as an empty one, BOS read as
 * signal — and none of them stop being true when they are one click away. What
 * they were doing open was burying the two controls that actually steer the
 * view, under a column of text, on top of the picture both of them describe.
 *
 * The trigger states the count, so a reader knows there is something to open
 * rather than having to open it to find out.
 */
function Caveats({ coverage, lit }: { coverage: string | null; lit: LitSet }) {
  const [open, setOpen] = useState(false)

  const sliced = lit.fired !== null && lit.shown < lit.fired
  const notes =
    (sliced ? 1 : 0) +
    (lit.unplaced.length > 0 ? 1 : 0) +
    (coverage ? 1 : 0) +
    (lit.bosExcluded ? 1 : 0)
  if (notes === 0) return null

  if (!open) {
    return (
      <button
        className="text-text-tertiary hover:text-text-primary pointer-events-auto font-mono transition-colors duration-150"
        onClick={() => setOpen(true)}
        type="button"
      >
        ▸ {notes} note{notes === 1 ? '' : 's'} on this set
      </button>
    )
  }

  return (
    <div className="space-y-2">
      <button
        className="text-text-tertiary hover:text-text-primary pointer-events-auto font-mono transition-colors duration-150"
        onClick={() => setOpen(false)}
        type="button"
      >
        ▾ {notes} note{notes === 1 ? '' : 's'} on this set
      </button>

      {sliced ? (
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
          <ul className="border-border-subtle bg-bg-elevated rounded-[2px] border">
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
 * Everything about the lit set that is not the count itself.
 *
 * How the scope combined a feature it met more than once, and which areas it
 * lit — both true, neither needed to read the picture. They were sitting open
 * under the count, and a reader who wanted to know how many features were
 * drawn had to walk past a four-hundred-row area list to find out.
 *
 * The area counts stay inside rather than being dropped: a bright area reads
 * as "all of this lit up" when it is often one member of four hundred, so the
 * count is the difference between a finding and an illusion. Folded, not
 * deleted — and the trigger says how many there are, so the reader knows
 * whether opening it is worth the click.
 */
function SetDetail({
  aggregation,
  areas,
}: {
  aggregation: LitSet['aggregation']
  areas: LitArea[]
}) {
  const [open, setOpen] = useState(false)

  if (aggregation === null && areas.length === 0) return null

  return (
    <div className="space-y-1">
      <button
        aria-expanded={open}
        className="text-text-tertiary hover:text-text-primary pointer-events-auto font-mono transition-colors duration-150"
        onClick={() => setOpen((value) => !value)}
        type="button"
      >
        {open ? '\u2212' : '+'} {areas.length === 0 ? 'detail' : `${areas.length} area${areas.length === 1 ? '' : 's'} lit`}
      </button>

      {open ? (
        <div className="space-y-1">
          {aggregation !== null ? (
            <div className="text-text-secondary flex justify-between gap-2 leading-4">
              <span>combined by</span>
              <span className="text-text-primary font-mono">{AGGREGATION}</span>
            </div>
          ) : null}
          <AreaReadout areas={areas} />
        </div>
      ) : null}
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
      <ul className="border-border-subtle bg-bg-elevated space-y-0.5 rounded-[2px] border px-2 py-1">
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
  onTogglePlay,
  playing,
  scope,
  selection,
  trace,
}: {
  featureLayers: number[]
  onSelectLayer?: (layer: number) => void
  onTogglePlay: () => void
  playing: boolean
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

  const step =
    'text-text-tertiary enabled:hover:bg-white/[0.02] disabled:text-text-disabled ' +
    'px-2 py-1 font-mono transition-colors duration-150 disabled:cursor-not-allowed'

  return (
    <div className="space-y-1">
      <div className="border-border-subtle bg-bg-elevated pointer-events-auto flex items-center rounded-[2px] border">
        <button
          aria-label="Previous layer"
          className={step}
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
          className={step}
          disabled={!canGoOn}
          onClick={() => onSelectLayer(layer + 1)}
          type="button"
        >
          ▶
        </button>
        <button
          aria-label={playing ? 'Stop the sweep' : 'Sweep every layer in order'}
          className={
            'border-border-subtle enabled:hover:bg-white/[0.02] disabled:text-text-disabled ' +
            'border-l px-2 py-1 font-mono transition-colors duration-150 disabled:cursor-not-allowed ' +
            (playing ? 'text-const' : 'text-text-tertiary')
          }
          disabled={inert}
          onClick={onTogglePlay}
          type="button"
        >
          {playing ? '■ stop' : '▶ sweep'}
        </button>
      </div>

      {/* What the sweep is doing, while it does it — including the part the
          picture cannot say for itself, which is that the blue is time and
          not a weaker activation. */}
      {playing ? (
        <p className="text-text-tertiary leading-4">
          Sweeping L0 → L{last}, one layer every {(SWEEP_MS / 1000).toFixed(2)}s. The last{' '}
          {TRAIL_DEPTH} layers stay on screen in{' '}
          <span className="text-fn">blue</span>, fading with age — that is order, not strength.
        </p>
      ) : null}

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
    <div className="border-border-subtle bg-bg-elevated absolute bottom-3 left-3 max-w-[20rem] space-y-1 rounded-[2px] border px-2.5 py-2 text-[11px]">
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
function NodeDetail({
  node,
  onUnpin,
  trace,
}: {
  node: LitNode
  /** Present only when this panel is pinned, which is what makes it dismissable. */
  onUnpin?: () => void
  trace: Trace | null
}) {
  const key = `${node.layer}/${node.feature}`
  const label = trace?.labels[key] ?? null
  const record = trace?.passes.find((p) => p.name === 'labels')
  const labelsRan = record !== undefined

  const pinned = onUnpin !== undefined

  return (
    <div
      className={
        'bg-bg-elevated absolute bottom-3 left-3 max-w-[20rem] space-y-1 rounded-[2px] ' +
        'border px-2.5 py-2 text-[11px] ' +
        // A pinned panel is a held state, so it says so with its border rather
        // than with a word: the reader needs to know it will not close on its
        // own, and the ring on the node is the other half of the same signal.
        (pinned ? 'border-border-strong' : 'border-border-subtle')
      }
    >
      {pinned ? (
        <button
          aria-label="Unpin this feature"
          className="text-text-tertiary hover:text-text-primary absolute top-1.5 right-2 font-mono transition-colors duration-150"
          onClick={onUnpin}
          type="button"
        >
          ✕
        </button>
      ) : null}

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

      <p className="text-text-disabled font-mono text-[10px] leading-4">
        {pinned ? 'pinned · esc or click away to close' : 'click to pin'}
      </p>
    </div>
  )
}

// --------------------------------------------------------------------------
// activity rendering
// --------------------------------------------------------------------------

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
  const trail = handles.trailCloud?.material as THREE.ShaderMaterial | undefined
  if (trail) trail.uniforms.detail.value = detail
  const areas = handles.areaCloud?.material as THREE.ShaderMaterial | undefined
  if (areas) areas.uniforms.detail.value = detail
  const pin = handles.pinMarker?.material as THREE.ShaderMaterial | undefined
  if (pin) pin.uniforms.detail.value = detail
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
    // A label on the far side of the cloud dims, so the near ones read first.
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
    <div className="border-border-subtle bg-bg-elevated text-text-tertiary pointer-events-none max-w-[15rem] rounded-[2px] border px-2.5 py-2 text-[11px] leading-4">
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
      className="border-border-strong bg-bg-elevated text-text-secondary pointer-events-none rounded-sm border px-2.5 py-1 font-mono text-[10px] tabular-nums"
    >
      <span aria-hidden="true" className="bg-fn mr-1.5 inline-block size-1.5 rounded-full" />
      {copy}
    </div>
  )
}

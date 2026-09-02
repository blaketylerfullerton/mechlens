import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js'
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js'
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js'
import { OutputPass } from 'three/examples/jsm/postprocessing/OutputPass.js'

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
  const material = new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.55 })
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

export function Brain() {
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const scene = new THREE.Scene()
    const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100)
    camera.position.set(0, 2, 4.6)
    camera.lookAt(0, -0.15, 0)

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false })
    renderer.setClearColor(0x000000, 1)
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    container.appendChild(renderer.domElement)

    const brainGroup = new THREE.Group()
    scene.add(brainGroup)

    const geometry = buildBrainGeometry()

    // Translucent glass shell — see-through so the underlying fold structure
    // (and eventually labeled cortex regions) reads clearly against black.
    const material = new THREE.MeshPhysicalMaterial({
      color: new THREE.Color(0xf2f2f2),
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
    // fissure, ...) traced over the glass so individual cortex regions can
    // be told apart and, later, labeled.
    const lobeBoundaries = buildLobeBoundaries()
    brainGroup.add(lobeBoundaries)

    // Brainstem: a short tapered stalk beneath the cerebellum, so the
    // silhouette reads as a brain rather than a bare cerebral mass.
    const brainstemGeometry = new THREE.CylinderGeometry(0.075, 0.12, 0.4, 16)
    const brainstemMesh = new THREE.Mesh(brainstemGeometry, material)
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

    const composer = new EffectComposer(renderer)
    composer.addPass(new RenderPass(scene, camera))
    const bloomPass = new UnrealBloomPass(new THREE.Vector2(1, 1), 0.15, 0.4, 0.85)
    composer.addPass(bloomPass)
    composer.addPass(new OutputPass())

    const resize = () => {
      const width = container.clientWidth
      const height = container.clientHeight
      camera.aspect = width / height
      camera.updateProjectionMatrix()
      renderer.setSize(width, height)
      composer.setSize(width, height)
      bloomPass.setSize(width, height)
    }
    resize()
    const resizeObserver = new ResizeObserver(resize)
    resizeObserver.observe(container)

    let frameId: number
    const clock = new THREE.Clock()
    const animate = () => {
      const elapsed = clock.getElapsedTime()
      brainGroup.rotation.y = elapsed * 0.04
      brainGroup.rotation.z = Math.sin(elapsed * 0.15) * 0.04
      brainGroup.position.y = Math.sin(elapsed * 0.3) * 0.06

      composer.render()
      frameId = requestAnimationFrame(animate)
    }
    animate()

    return () => {
      cancelAnimationFrame(frameId)
      resizeObserver.disconnect()
      container.removeChild(renderer.domElement)
      geometry.dispose()
      brainstemGeometry.dispose()
      material.dispose()
      lobeBoundaries.children.forEach((line) => {
        ;(line as THREE.Line).geometry.dispose()
        ;((line as THREE.Line).material as THREE.Material).dispose()
      })
      renderer.dispose()
      composer.dispose()
    }
  }, [])

  return <div ref={containerRef} className="h-full w-full" />
}

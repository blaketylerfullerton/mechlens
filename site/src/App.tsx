import { IconBrandGithub } from '@tabler/icons-react';
import { NeuroNoise } from '@paper-design/shaders-react';

/**
 * The mechlens landing page.
 *
 * Deliberately one file and plain markup. Everything sits inside one centred
 * column (`Container`) so the page reads as a document rather than content
 * pinned to the corner of a wide screen.
 *
 * The design tokens are the same ones the app uses (`src/index.css`), so
 * anything built for `frontend/` renders identically here.
 */

function App() {
  return (
    <div className="relative isolate min-h-screen bg-bg-base text-text-primary">
      <PageBackdrop />

      <Nav />
      
      <main>
        <Hero />
       
      </main>
      <Footer />
    </div>
  )
}

/**
 * Animated noise behind the whole page. `fixed` so it covers the viewport and
 * stays put while you scroll; no pointer events, and faded at the edges so the
 * text stays the brightest thing on screen.
 */
function PageBackdrop() {
  return (
    <div
      aria-hidden
      className="pointer-events-none fixed inset-0 -z-10 opacity-70 [mask-image:radial-gradient(75%_70%_at_35%_35%,black,transparent)]"
    >
      <NeuroNoise
        width="100%"
        height="100%"
        colorFront="#ffffff"
        colorMid="#4c749a"
        colorBack="#0a0b0d"
        brightness={0.05}
        contrast={0.3}
        speed={0.38}
        scale={1.52}
      />
    </div>
  )
}

function Nav() {
  return (
    <nav className="sticky top-0 z-10 border-b border-border-subtle bg-bg-base/70 backdrop-blur-md">
      <Container className="flex h-14 items-center justify-between">
        <span className="font-mono text-sm text-text-primary">mechlens</span>
        <a
          className="inline-flex items-center gap-2 font-mono text-xs text-text-secondary transition-colors hover:text-fn"
          href="https://github.com/blaketylerfullerton/mechlens"
        >
          <IconBrandGithub className="size-4" stroke={1.5} />
          GitHub
        </a>
      </Container>
    </nav>
  )
}

function Hero() {
  return (
    <Container className="enter flex min-h-[calc(100svh-3.5rem)] flex-col justify-center py-24">
      <p className="mb-6 inline-flex w-fit items-center gap-2 rounded-full border border-border-subtle bg-bg-surface/60 px-3 py-1 font-mono text-xs tracking-wide text-text-secondary backdrop-blur-sm">
        <span className="size-1.5 rounded-full bg-fn" />
        mechlens
      </p>

      <h1 className="max-w-3xl text-4xl leading-tight font-medium tracking-tight sm:text-5xl">
        Look inside a language model{' '}
        <span className="text-text-secondary">on your own machine.</span>
      </h1>

      <p className="mt-6 max-w-2xl text-base leading-relaxed text-text-secondary">
        Send a prompt and see which of the model's{' '}
        <span className="font-mono text-text-primary">98,210</span> named
        features fired, and how the answer took shape layer by layer. Nothing
        leaves your machine.
      </p>

      <div className="mt-10 flex flex-wrap items-center gap-3">
        <a
          className="rounded-md bg-primary px-4 py-2 font-mono text-sm text-primary-foreground transition-opacity hover:opacity-90"
          href="#setup"
        >
          See it run
        </a>
        <a
          className="inline-flex items-center gap-2 rounded-md border border-border-strong px-4 py-2 font-mono text-sm text-text-secondary transition-colors hover:border-fn hover:text-text-primary"
          href="https://github.com/blaketylerfullerton/mechlens"
        >
          <IconBrandGithub className="size-4" stroke={1.5} />
          Clone the repo
        </a>
      </div>

      <p className="mt-5 font-mono text-xs text-text-tertiary">
        gemma-2-2b · CUDA · Apache-2.0
      </p>
    </Container>
  )
}



function Footer() {
  return (
    <footer className="border-t border-border-subtle">
      <Container className="flex h-16 items-center justify-between font-mono text-xs text-text-tertiary">
        <span>mechlens · Apache-2.0</span>
        <a
          className="transition-colors hover:text-fn"
          href="https://github.com/blaketylerfullerton/mechlens"
        >
          GitHub
        </a>
      </Container>
    </footer>
  )
}

/* -------------------------------------------------------------------------
   Small layout helpers. Nothing clever — replace them freely.
   ------------------------------------------------------------------------- */

/** The one thing keeping the page from sprawling on a wide screen. */
function Container({
  children,
  className = '',
}: {
  children: React.ReactNode
  className?: string
}) {
  return (
    <div className={`mx-auto w-full max-w-5xl px-6 ${className}`}>
      {children}
    </div>
  )
}





export default App

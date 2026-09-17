import { useEffect, useMemo, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'
import 'katex/dist/katex.min.css'
import './home.css'
import './how.css'
import source from './content/how.md?raw'
import { SiteFooter, SiteHeader } from './SiteChrome'


/** `## 01 · Get the activations` -> a slug the contents rail and the URL share. */
function slugify(text: string) {
  return text.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '')
}

function headings(markdown: string) {
  return markdown
    .split('\n')
    .filter(line => line.startsWith('## '))
    .map(line => line.slice(3).trim())
    .map(title => ({ title, id: slugify(title) }))
}

export default function HowPage() {
  const sections = useMemo(() => headings(source), [])
  const [active, setActive] = useState(sections[0]?.id ?? '')

  // Mark the section being read. IntersectionObserver rather than a scroll
  // handler so nothing runs on frames where nothing crossed.
  useEffect(() => {
    const targets = sections
      .map(({ id }) => document.getElementById(id))
      .filter((el): el is HTMLElement => el !== null)
    if (!targets.length) return
    const observer = new IntersectionObserver(
      entries => {
        const visible = entries.filter(e => e.isIntersecting)
        if (visible.length) setActive(visible[0].target.id)
      },
      { rootMargin: '-10% 0px -70% 0px', threshold: 0 },
    )
    targets.forEach(el => observer.observe(el))
    return () => observer.disconnect()
  }, [sections])

  return (
    <div className="homepage how-page">
      <a className="skip-link" href="#main">Skip to content</a>
      <SiteHeader current="how" />

      <main id="main">
        <section className="container how-hero">
          <div className="enter">
            <p className="eyebrow">Method</p>
            <h1>How it works.</h1>
            <p className="how-lead">
              From a prompt to a picture, in five steps — each one written as the
              formula the code computes, and the measurement that catches it
              going wrong.
            </p>
            <p className="model-meta">
              gemma-2-2b <span>/</span> 26 layers <span>/</span> d_model 2304 <span>/</span> Gemma Scope 16k
            </p>
          </div>
        </section>

        <div className="container how-body">
          <nav className="how-contents" aria-label="Contents">
            <p className="small-label">Contents</p>
            <ol>
              {sections.map(({ title, id }) => (
                <li key={id}>
                  <a href={`#${id}`} className={active === id ? 'active' : undefined}>{title}</a>
                </li>
              ))}
            </ol>
          </nav>

          <article className="prose">
            <ReactMarkdown
              remarkPlugins={[remarkMath]}
              rehypePlugins={[rehypeKatex]}
              components={{
                h2: ({ children }) => {
                  const text = String(children)
                  return <h2 id={slugify(text)}>{children}</h2>
                },
              }}
            >
              {source}
            </ReactMarkdown>
          </article>
        </div>
      </main>

      <SiteFooter />
    </div>
  )
}

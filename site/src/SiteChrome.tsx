import { IconBrandGithub } from '@tabler/icons-react'
import { navigate } from './lib/routing'

export const REPO = 'https://github.com/blaketylerfullerton/mechlens'

// Header and footer shared by / and /how. Links to a section on the homepage
// go through navigate() so they work from either route.
export function SiteHeader({ current }: { current?: 'how' }) {
  return (
    <header className="site-header">
      <nav className="container nav" aria-label="Main navigation">
        <a href="/" className="wordmark" onClick={navigate('/')} aria-label="Mechlens home">
          <img src="/logo.svg" alt="" width={26} height={26} />mechlens
        </a>
        <div className="nav-links">
          <a href="/how" onClick={navigate('/how')} aria-current={current === 'how' ? 'page' : undefined}>How</a>
          <a href="/#setup" onClick={navigate('/#setup')}>Setup</a>
          <a href={REPO} className="nav-repo"><IconBrandGithub size={16} stroke={1.5} />GitHub</a>
        </div>
      </nav>
    </header>
  )
}

export function SiteFooter() {
  return (
    <footer className="site-footer">
      <div className="container footer">
        <div className="footer-brand">
          <span className="wordmark"><img src="/logo.svg" alt="" width={22} height={22} />mechlens</span>
          <p>Open source, open to inspection.</p>
        </div>
        <nav className="footer-links" aria-label="Footer">
          <a href="/how" onClick={navigate('/how')}>How it works</a>
          <a href="/#setup" onClick={navigate('/#setup')}>Setup</a>
          <a href={REPO}>GitHub</a>
          <a href={`${REPO}/blob/main/LICENSE`}>Apache-2.0</a>
        </nav>
        <p className="footer-meta machine"><span>gemma-2-2b</span><span>gemma-scope-2b-pt-res-canonical</span><span>hook_resid_post</span></p>
      </div>
    </footer>
  )
}

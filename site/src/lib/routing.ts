/** Two pages, so two lines of routing rather than a router dependency. */

export type Route = '/' | '/how'

export function currentRoute(): Route {
  return window.location.pathname.replace(/\/+$/, '') === '/how' ? '/how' : '/'
}

/** Client-side link handler: keeps modified clicks as real navigations. */
export function navigate(to: string) {
  return (event: React.MouseEvent<HTMLAnchorElement>) => {
    if (event.defaultPrevented || event.button !== 0) return
    if (event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return
    event.preventDefault()
    const [path, hash] = to.split('#')
    window.history.pushState({}, '', to)
    window.dispatchEvent(new PopStateEvent('popstate'))
    if (hash) {
      // Let the target page mount before trying to reach an anchor inside it.
      requestAnimationFrame(() => document.getElementById(hash)?.scrollIntoView())
    } else if (path) {
      window.scrollTo(0, 0)
    }
  }
}

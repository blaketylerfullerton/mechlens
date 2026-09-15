import { StrictMode, useEffect, useState } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import HowPage from './HowPage.tsx'
import { currentRoute, type Route } from './lib/routing'

function Site() {
  const [route, setRoute] = useState<Route>(currentRoute)
  useEffect(() => {
    const onPop = () => setRoute(currentRoute())
    window.addEventListener('popstate', onPop)
    return () => window.removeEventListener('popstate', onPop)
  }, [])
  return route === '/how' ? <HowPage /> : <App />
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <Site />
  </StrictMode>,
)

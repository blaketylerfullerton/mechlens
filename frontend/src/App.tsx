import { Brain } from '@/components/Brain'
import { ChatPanel } from '@/components/ChatPanel'

function App() {
  return (
    <div className="relative h-svh w-full bg-background">
      <div className="absolute inset-0">
        <Brain />
      </div>
      <ChatPanel />
    </div>
  )
}

export default App

import { ChatPanel } from '@/components/ChatPanel'
import { TraceViewer } from '@/components/TraceViewer'
import { useTrace } from '@/hooks/useTrace'

function App() {
  const { error, run, status, trace } = useTrace()

  return (
    <div className="min-h-svh bg-[radial-gradient(circle_at_top,_#17304a_0%,_#0a1220_34%,_#060912_72%)] text-slate-100">
      <TraceViewer error={error} status={status} trace={trace} />
      <ChatPanel error={error} onTraceRequest={run} status={status} />
    </div>
  )
}

export default App

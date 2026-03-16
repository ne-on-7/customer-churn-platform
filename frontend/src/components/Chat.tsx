import { useState, useRef, useEffect } from 'react'
import { Send, Bot, User, Loader2 } from 'lucide-react'
import { streamChat } from '../lib/api'
import type { ChatMessage, ToolResult } from '../types'

function ToolResultCard({ result }: { result: ToolResult }) {
  const { tool, result: data } = result

  if (tool === 'predict_churn' && data.churn_probability !== undefined) {
    const prob = data.churn_probability as number
    const tier = data.risk_tier as string
    const reasons = (data.top_reasons || []) as string[]
    const tierClasses: Record<string, string> = {
      High: 'text-red-500',
      Medium: 'text-amber-500',
      Low: 'text-emerald-500',
    }
    const colorClass = tierClasses[tier] || 'text-gray-500'
    return (
      <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-4 my-2">
        <div className="flex items-center gap-4">
          <div className="text-center">
            <div className={`text-3xl font-bold ${colorClass}`}>{(prob * 100).toFixed(1)}%</div>
            <div className={`text-xs font-medium ${colorClass} uppercase mt-1`}>{tier} Risk</div>
          </div>
          <div className="flex-1 border-l border-gray-200 dark:border-gray-700 pl-4">
            <p className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">Top Churn Drivers</p>
            {reasons.map((r, i) => (
              <p key={i} className="text-sm text-gray-700 dark:text-gray-300">{i + 1}. {r}</p>
            ))}
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="bg-gray-50 dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-3 my-2 text-xs">
      <p className="font-medium text-gray-500 mb-1">{tool}</p>
      <pre className="text-gray-600 dark:text-gray-400 overflow-x-auto">{JSON.stringify(data, null, 2)}</pre>
    </div>
  )
}

export default function Chat() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [streaming, setStreaming] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)
  const abortRef = useRef<(() => void) | null>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const sendMessage = (text: string) => {
    if (!text.trim() || streaming) return
    setInput('')

    const userMsg: ChatMessage = { role: 'user', content: text.trim() }
    const newMessages = [...messages, userMsg]
    setMessages([...newMessages, { role: 'assistant', content: '', toolResults: [] }])
    setStreaming(true)

    const apiMessages = newMessages.map(({ role, content }) => ({ role, content }))

    abortRef.current = streamChat(
      apiMessages,
      (data) => {
        setMessages((prev) => {
          const updated = [...prev]
          const last = { ...updated[updated.length - 1] }
          if (data.type === 'text_delta' && data.content) {
            last.content += data.content
          } else if (data.type === 'tool_result' && data.tool && data.result) {
            last.toolResults = [...(last.toolResults || []), { tool: data.tool, result: data.result }]
          }
          updated[updated.length - 1] = last
          return updated
        })
      },
      () => setStreaming(false),
      (err) => {
        console.error(err)
        setMessages((prev) => {
          const updated = [...prev]
          const last = { ...updated[updated.length - 1] }
          last.content += '\n\n*Error: Could not get a response. Please try again.*'
          updated[updated.length - 1] = last
          return updated
        })
        setStreaming(false)
      }
    )
  }

  const send = () => sendMessage(input)

  return (
    <div className="flex flex-col h-full">
      {/* Header */}
      <div className="border-b border-gray-200 dark:border-gray-800 px-6 py-4">
        <h1 className="text-xl font-semibold">AI Churn Assistant</h1>
        <p className="text-sm text-gray-500 dark:text-gray-400 mt-0.5">
          Ask questions about customer churn, run predictions, or analyze business impact
        </p>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
        {messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full text-center">
            <Bot size={48} className="text-indigo-400 mb-4" />
            <h2 className="text-lg font-medium text-gray-700 dark:text-gray-300">How can I help?</h2>
            <p className="text-sm text-gray-500 dark:text-gray-400 mt-1 max-w-md">
              Try asking about churn predictions, model performance, or business impact analysis.
            </p>
            <div className="flex flex-wrap gap-2 mt-6 max-w-lg justify-center">
              {[
                "What's the churn risk for a senior citizen on month-to-month with fiber optic?",
                'Which model performs best and why?',
                'What is the ROI of our retention program?',
                'Show me high-risk customers',
              ].map((q) => (
                <button
                  key={q}
                  onClick={() => sendMessage(q)}
                  className="text-xs px-3 py-2 rounded-lg bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400 hover:bg-indigo-50 dark:hover:bg-indigo-900/20 hover:text-indigo-600 dark:hover:text-indigo-400 transition-colors text-left"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg, i) => (
          <div key={i} className={`flex gap-3 animate-fade-in ${msg.role === 'user' ? 'justify-end' : ''}`}>
            {msg.role === 'assistant' && (
              <div className="w-8 h-8 rounded-lg bg-indigo-100 dark:bg-indigo-900/30 flex items-center justify-center flex-shrink-0">
                <Bot size={16} className="text-indigo-600 dark:text-indigo-400" />
              </div>
            )}
            <div
              className={`max-w-2xl rounded-2xl px-4 py-3 text-sm leading-relaxed ${
                msg.role === 'user'
                  ? 'bg-indigo-500 text-white'
                  : 'bg-gray-100 dark:bg-gray-800 text-gray-800 dark:text-gray-200'
              }`}
            >
              <div className="whitespace-pre-wrap">{msg.content}</div>
              {msg.toolResults?.map((tr, j) => <ToolResultCard key={j} result={tr} />)}
              {msg.role === 'assistant' && streaming && i === messages.length - 1 && !msg.content && (
                <Loader2 size={16} className="animate-spin text-gray-400" />
              )}
            </div>
            {msg.role === 'user' && (
              <div className="w-8 h-8 rounded-lg bg-indigo-500 flex items-center justify-center flex-shrink-0">
                <User size={16} className="text-white" />
              </div>
            )}
          </div>
        ))}
        <div ref={bottomRef} />
      </div>

      {/* Input */}
      <div className="border-t border-gray-200 dark:border-gray-800 px-6 py-4">
        <div className="flex gap-3 max-w-3xl mx-auto">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && send()}
            placeholder="Ask about customer churn..."
            className="flex-1 px-4 py-3 rounded-xl border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent"
            disabled={streaming}
          />
          <button
            onClick={send}
            disabled={streaming || !input.trim()}
            className="px-4 py-3 rounded-xl bg-indigo-500 hover:bg-indigo-600 disabled:opacity-50 disabled:cursor-not-allowed text-white transition-colors"
          >
            {streaming ? <Loader2 size={18} className="animate-spin" /> : <Send size={18} />}
          </button>
        </div>
      </div>
    </div>
  )
}

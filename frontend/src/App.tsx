import { FormEvent, ReactNode, useEffect, useMemo, useRef, useState } from "react"
import { AlertCircle, Factory, Loader2, Send, ShieldCheck } from "lucide-react"

import { Button } from "@/components/ui/button"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Textarea } from "@/components/ui/textarea"
import { cn } from "@/lib/utils"

type Role = "worker" | "help"

type ChatMessage = {
  id: string
  role: Role
  content: string
}

type ParsedSseEvent = {
  event: string
  data: unknown
}

const THREAD_STORAGE_KEY = "manufacturing-help-desk-conversation-id"

const starterMessages: ChatMessage[] = [
  {
    id: "welcome",
    role: "help",
    content:
      "Ask a floor question about safety, maintenance, or quality. I will point you to the right information and give a clear answer.",
  },
]

async function readSseStream(
  stream: ReadableStream<Uint8Array>,
  onEvent: (event: ParsedSseEvent) => void,
) {
  const reader = stream.getReader()
  const decoder = new TextDecoder()
  let buffer = ""

  while (true) {
    const { done, value } = await reader.read()

    if (done) {
      break
    }

    buffer += decoder.decode(value, { stream: true })
    buffer = flushSseEvents(buffer, onEvent)
  }

  buffer += decoder.decode()

  if (buffer.trim()) {
    handleSseBlock(buffer, onEvent)
  }
}

function flushSseEvents(buffer: string, onEvent: (event: ParsedSseEvent) => void) {
  let nextBuffer = buffer
  let boundaryIndex = nextBuffer.indexOf("\n\n")

  while (boundaryIndex !== -1) {
    const block = nextBuffer.slice(0, boundaryIndex)
    nextBuffer = nextBuffer.slice(boundaryIndex + 2)
    handleSseBlock(block, onEvent)
    boundaryIndex = nextBuffer.indexOf("\n\n")
  }

  return nextBuffer
}

function handleSseBlock(block: string, onEvent: (event: ParsedSseEvent) => void) {
  const parsed = parseSseBlock(block)

  if (parsed) {
    onEvent(parsed)
  }
}

function parseSseBlock(block: string): ParsedSseEvent | null {
  let event = "message"
  const dataLines: string[] = []

  for (const line of block.split(/\r?\n/)) {
    if (line.startsWith("event:")) {
      event = line.slice("event:".length).trim()
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice("data:".length).trimStart())
    }
  }

  if (dataLines.length === 0) {
    return null
  }

  return {
    event,
    data: JSON.parse(dataLines.join("\n")) as unknown,
  }
}

function threadIdFromPayload(payload: unknown) {
  if (isRecord(payload) && typeof payload.thread_id === "string") {
    return payload.thread_id
  }

  return null
}

function tokenTextFromPayload(payload: unknown) {
  if (isRecord(payload) && typeof payload.text === "string") {
    return payload.text
  }

  return ""
}

function errorMessageFromPayload(payload: unknown) {
  if (isRecord(payload) && typeof payload.message === "string" && payload.message) {
    return payload.message
  }

  return "Unable to stream the help desk response."
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null
}

function MarkdownContent({ content }: { content: string }) {
  const blocks = parseMarkdownBlocks(content)

  return (
    <div className="space-y-3">
      {blocks.map((block, index) => {
        const key = `${block.type}-${index}`

        if (block.type === "heading") {
          const HeadingTag = `h${block.level}` as const

          return (
            <HeadingTag key={key} className="text-base font-semibold leading-6 text-zinc-50">
              {renderInlineMarkdown(block.text, key)}
            </HeadingTag>
          )
        }

        if (block.type === "unordered-list") {
          return (
            <ul key={key} className="list-disc space-y-1 pl-5">
              {block.items.map((item, itemIndex) => (
                <li key={`${key}-${itemIndex}`}>{renderInlineMarkdown(item, `${key}-${itemIndex}`)}</li>
              ))}
            </ul>
          )
        }

        if (block.type === "ordered-list") {
          return (
            <ol key={key} className="list-decimal space-y-1 pl-5">
              {block.items.map((item, itemIndex) => (
                <li key={`${key}-${itemIndex}`}>{renderInlineMarkdown(item, `${key}-${itemIndex}`)}</li>
              ))}
            </ol>
          )
        }

        if (block.type === "blockquote") {
          return (
            <blockquote key={key} className="border-l-2 border-zinc-700 pl-3 text-zinc-300">
              {renderInlineMarkdown(block.text, key)}
            </blockquote>
          )
        }

        if (block.type === "code") {
          return (
            <pre key={key} className="overflow-x-auto rounded-md bg-zinc-900 px-3 py-2 text-xs leading-5 text-zinc-100">
              <code>{block.text}</code>
            </pre>
          )
        }

        return (
          <p key={key} className="whitespace-pre-wrap">
            {renderInlineMarkdown(block.text, key)}
          </p>
        )
      })}
    </div>
  )
}

type MarkdownBlock =
  | { type: "paragraph"; text: string }
  | { type: "heading"; level: 1 | 2 | 3 | 4 | 5 | 6; text: string }
  | { type: "unordered-list"; items: string[] }
  | { type: "ordered-list"; items: string[] }
  | { type: "blockquote"; text: string }
  | { type: "code"; text: string }

function parseMarkdownBlocks(content: string): MarkdownBlock[] {
  const lines = content.replace(/\r\n/g, "\n").split("\n")
  const blocks: MarkdownBlock[] = []
  let index = 0

  while (index < lines.length) {
    const line = lines[index]

    if (!line.trim()) {
      index += 1
      continue
    }

    if (line.startsWith("```")) {
      const codeLines: string[] = []
      index += 1

      while (index < lines.length && !lines[index].startsWith("```")) {
        codeLines.push(lines[index])
        index += 1
      }

      blocks.push({ type: "code", text: codeLines.join("\n") })
      index += index < lines.length ? 1 : 0
      continue
    }

    const heading = /^(#{1,6})\s+(.+)$/.exec(line)

    if (heading) {
      blocks.push({
        type: "heading",
        level: heading[1].length as 1 | 2 | 3 | 4 | 5 | 6,
        text: heading[2],
      })
      index += 1
      continue
    }

    if (/^\s*[-*]\s+/.test(line)) {
      const items: string[] = []

      while (index < lines.length && /^\s*[-*]\s+/.test(lines[index])) {
        items.push(lines[index].replace(/^\s*[-*]\s+/, ""))
        index += 1
      }

      blocks.push({ type: "unordered-list", items })
      continue
    }

    if (/^\s*\d+\.\s+/.test(line)) {
      const items: string[] = []

      while (index < lines.length && /^\s*\d+\.\s+/.test(lines[index])) {
        items.push(lines[index].replace(/^\s*\d+\.\s+/, ""))
        index += 1
      }

      blocks.push({ type: "ordered-list", items })
      continue
    }

    if (/^\s*>\s?/.test(line)) {
      const quoteLines: string[] = []

      while (index < lines.length && /^\s*>\s?/.test(lines[index])) {
        quoteLines.push(lines[index].replace(/^\s*>\s?/, ""))
        index += 1
      }

      blocks.push({ type: "blockquote", text: quoteLines.join("\n") })
      continue
    }

    const paragraphLines: string[] = []

    while (index < lines.length && lines[index].trim() && !isMarkdownBlockStart(lines[index])) {
      paragraphLines.push(lines[index])
      index += 1
    }

    blocks.push({ type: "paragraph", text: paragraphLines.join("\n") })
  }

  return blocks
}

function isMarkdownBlockStart(line: string) {
  return (
    line.startsWith("```") ||
    /^(#{1,6})\s+/.test(line) ||
    /^\s*[-*]\s+/.test(line) ||
    /^\s*\d+\.\s+/.test(line) ||
    /^\s*>\s?/.test(line)
  )
}

function renderInlineMarkdown(text: string, keyPrefix: string): ReactNode[] {
  const nodes: ReactNode[] = []
  let cursor = 0

  while (cursor < text.length) {
    const codeEnd = text[cursor] === "`" ? text.indexOf("`", cursor + 1) : -1

    if (codeEnd !== -1) {
      nodes.push(
        <code key={`${keyPrefix}-code-${cursor}`} className="rounded bg-zinc-800 px-1 py-0.5 text-[0.85em] text-zinc-100">
          {text.slice(cursor + 1, codeEnd)}
        </code>,
      )
      cursor = codeEnd + 1
      continue
    }

    const linkMatch = text.slice(cursor).match(/^\[([^\]]+)]\((https?:\/\/[^)\s]+|mailto:[^)\s]+)\)/)

    if (linkMatch) {
      nodes.push(
        <a
          key={`${keyPrefix}-link-${cursor}`}
          className="font-medium text-emerald-300 underline underline-offset-2 hover:text-emerald-200"
          href={linkMatch[2]}
          rel="noreferrer"
          target="_blank"
        >
          {renderInlineMarkdown(linkMatch[1], `${keyPrefix}-link-${cursor}`)}
        </a>,
      )
      cursor += linkMatch[0].length
      continue
    }

    const strongDelimiter = text.startsWith("**", cursor) ? "**" : text.startsWith("__", cursor) ? "__" : null

    if (strongDelimiter) {
      const strongEnd = text.indexOf(strongDelimiter, cursor + 2)

      if (strongEnd !== -1) {
        nodes.push(
          <strong key={`${keyPrefix}-strong-${cursor}`} className="font-semibold text-zinc-50">
            {renderInlineMarkdown(text.slice(cursor + 2, strongEnd), `${keyPrefix}-strong-${cursor}`)}
          </strong>,
        )
        cursor = strongEnd + 2
        continue
      }
    }

    const emphasisDelimiter =
      text[cursor] === "*" && text[cursor + 1] !== "*" ? "*" : text[cursor] === "_" && text[cursor + 1] !== "_" ? "_" : null

    if (emphasisDelimiter) {
      const emphasisEnd = text.indexOf(emphasisDelimiter, cursor + 1)

      if (emphasisEnd !== -1) {
        nodes.push(
          <em key={`${keyPrefix}-em-${cursor}`} className="italic">
            {renderInlineMarkdown(text.slice(cursor + 1, emphasisEnd), `${keyPrefix}-em-${cursor}`)}
          </em>,
        )
        cursor = emphasisEnd + 1
        continue
      }
    }

    const nextSpecial = findNextInlineSpecial(text, cursor + 1)
    nodes.push(text.slice(cursor, nextSpecial))
    cursor = nextSpecial
  }

  return nodes
}

function findNextInlineSpecial(text: string, start: number) {
  const indexes = ["`", "[", "*", "_"]
    .map((marker) => text.indexOf(marker, start))
    .filter((index) => index !== -1)

  return indexes.length > 0 ? Math.min(...indexes) : text.length
}

function App() {
  const [messages, setMessages] = useState<ChatMessage[]>(starterMessages)
  const [draft, setDraft] = useState("")
  const [threadId, setThreadId] = useState<string | null>(() => localStorage.getItem(THREAD_STORAGE_KEY))
  const [isSending, setIsSending] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const messagesEndRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({
      behavior: "smooth",
      block: "end",
    })
  }, [messages, isSending])

  const canSend = useMemo(() => draft.trim().length > 0 && !isSending, [draft, isSending])

  async function submitMessage(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const message = draft.trim()

    if (!message || isSending) {
      return
    }

    const workerMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: "worker",
      content: message,
    }
    const helpMessageId = crypto.randomUUID()

    setMessages((current) => [
      ...current,
      workerMessage,
      {
        id: helpMessageId,
        role: "help",
        content: "",
      },
    ])
    setDraft("")
    setError(null)
    setIsSending(true)

    try {
      const response = await fetch("/api/chat/stream", {
        method: "POST",
        headers: {
          Accept: "text/event-stream",
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message,
          thread_id: threadId,
        }),
      })

      if (!response.ok) {
        throw new Error(`Request failed with status ${response.status}`)
      }

      if (!response.body) {
        throw new Error("The help desk did not return a response stream.")
      }

      await readSseStream(response.body, ({ event, data }) => {
        if (event === "thread" || event === "done") {
          const nextThreadId = threadIdFromPayload(data)

          if (nextThreadId) {
            localStorage.setItem(THREAD_STORAGE_KEY, nextThreadId)
            setThreadId(nextThreadId)
          }

          return
        }

        if (event === "token") {
          const tokenText = tokenTextFromPayload(data)

          if (tokenText) {
            setMessages((current) =>
              current.map((currentMessage) =>
                currentMessage.id === helpMessageId
                  ? { ...currentMessage, content: currentMessage.content + tokenText }
                  : currentMessage,
              ),
            )
          }

          return
        }

        if (event === "error") {
          throw new Error(errorMessageFromPayload(data))
        }
      })
    } catch (caught) {
      const message = caught instanceof Error ? caught.message : "Unable to reach the help desk."
      setMessages((current) =>
        current.filter((currentMessage) => currentMessage.id !== helpMessageId || currentMessage.content.length > 0),
      )
      setError(message)
    } finally {
      setIsSending(false)
    }
  }

  return (
    <main className="min-h-svh bg-zinc-950 text-zinc-50">
      <div className="mx-auto flex min-h-svh w-full max-w-5xl flex-col px-4 py-5 sm:px-6 lg:px-8">
        <header className="flex flex-col gap-4 border-b border-zinc-800 pb-5 sm:flex-row sm:items-center sm:justify-between">
          <div className="flex items-center gap-3">
            <div className="flex size-11 items-center justify-center rounded-md bg-emerald-500 text-zinc-950">
              <Factory className="size-6" aria-hidden="true" />
            </div>
            <div>
              <h1 className="text-xl font-semibold tracking-normal">Manufacturing Help Desk</h1>
              <p className="text-sm text-zinc-400">Plain answers from your plant documents</p>
            </div>
          </div>
          <div className="flex items-center gap-2 rounded-md border border-zinc-800 px-3 py-2 text-xs text-zinc-300">
            <ShieldCheck className="size-4 text-emerald-400" aria-hidden="true" />
            <span>{threadId ? "Conversation saved" : "New conversation"}</span>
          </div>
        </header>

        <section className="grid flex-1 gap-5 py-5 lg:grid-cols-[220px_minmax(0,1fr)]">
          <aside className="hidden border-r border-zinc-800 pr-5 text-sm text-zinc-400 lg:block">
            <div className="space-y-4">
              <div>
                <h2 className="mb-2 text-sm font-medium text-zinc-100">I Can Help With</h2>
                <ul className="space-y-2">
                  <li>Safety procedures</li>
                  <li>Maintenance manuals</li>
                  <li>Quality standards</li>
                </ul>
              </div>
              <div className="rounded-md border border-zinc-800 bg-zinc-900/60 p-3">
                More detailed answers and follow-up help will be added in the next update.
              </div>
            </div>
          </aside>

          <div className="flex min-h-[calc(100svh-9rem)] flex-col overflow-hidden rounded-md border border-zinc-800 bg-zinc-900">
            <ScrollArea className="flex-1">
              <div className="space-y-4 p-4 sm:p-5">
                {messages.map((message) => (
                  <article
                    key={message.id}
                    className={cn(
                      "max-w-[88%] rounded-md px-4 py-3 text-sm leading-6",
                      message.role === "worker"
                        ? "ml-auto bg-emerald-500 text-zinc-950"
                        : "border border-zinc-800 bg-zinc-950 text-zinc-100",
                    )}
                  >
                    <div className="mb-1 text-xs font-medium uppercase tracking-normal opacity-70">
                      {message.role === "worker" ? "You" : "Plant Help"}
                    </div>
                    {message.role === "worker" ? (
                      <p className="whitespace-pre-wrap">{message.content}</p>
                    ) : (
                      <MarkdownContent content={message.content} />
                    )}
                  </article>
                ))}

                {isSending ? (
                  <div className="flex items-center gap-2 text-sm text-zinc-400">
                    <Loader2 className="size-4 animate-spin" aria-hidden="true" />
                    Checking the request.
                  </div>
                ) : null}
                <div ref={messagesEndRef} />
              </div>
            </ScrollArea>

            {error ? (
              <div className="mx-4 mb-3 flex items-center gap-2 rounded-md border border-red-900 bg-red-950 px-3 py-2 text-sm text-red-100 sm:mx-5">
                <AlertCircle className="size-4 shrink-0" aria-hidden="true" />
                <span>{error}</span>
              </div>
            ) : null}

            <form className="border-t border-zinc-800 p-4 sm:p-5" onSubmit={submitMessage}>
              <div className="flex gap-3">
                <Textarea
                  aria-label="Message"
                  className="min-h-11 resize-none border-zinc-700 bg-zinc-950 text-zinc-50 placeholder:text-zinc-500"
                  disabled={isSending}
                  onChange={(event) => setDraft(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter" && !event.shiftKey) {
                      event.preventDefault()
                      event.currentTarget.form?.requestSubmit()
                    }
                  }}
                  placeholder="Example: What should I check for a pump vibration alarm?"
                  value={draft}
                />
                <Button
                  aria-label="Send message"
                  className="h-11 w-11 shrink-0 bg-emerald-500 text-zinc-950 hover:bg-emerald-400"
                  disabled={!canSend}
                  size="icon"
                  type="submit"
                >
                  {isSending ? <Loader2 className="size-4 animate-spin" /> : <Send className="size-4" />}
                </Button>
              </div>
            </form>
          </div>
        </section>
      </div>
    </main>
  )
}

export default App

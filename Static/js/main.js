let queryHistory = []
let uploadPollRun = 0
let statusClearTimer = null
const STORAGE_KEYS = {
chat: "ragAssistant.chatMessages",
history: "ragAssistant.queryHistory"
}

function chatBox() {
return document.getElementById("chat")
}

function scrollChatToBottom() {
const chat = chatBox()
chat.scrollTop = chat.scrollHeight
}

function createMessage(type, text = "", allowHtml = false) {
const chat = chatBox()
const msg = document.createElement("div")
msg.className = "message " + type

const body = document.createElement("div")
body.className = "message-body"
if (allowHtml) {
body.innerHTML = text
} else {
body.textContent = text
}

msg.appendChild(body)
chat.appendChild(msg)
scrollChatToBottom()
return msg
}

function createUserMessage(queryItem) {
const msg = createMessage("user", queryItem.text)
msg.dataset.queryId = queryItem.id
return msg
}

function renderChartCanvas(container, chartData, chartType) {
if (!container || !Array.isArray(chartData) || !chartData.length || typeof Chart === "undefined") {
return
}

if (container._chartInstance) {
container._chartInstance.destroy()
container._chartInstance = null
}

container.innerHTML = ""
const canvas = document.createElement("canvas")
canvas.className = "chart-image"
container.appendChild(canvas)

const labels = chartData.map(item => String(item.label ?? ""))
const values = chartData.map(item => Number(item.value ?? 0))
const palette = ["#2f7db6", "#ef6c4f", "#4f9d69", "#8c5fbf", "#d8a31a", "#c94f7c"]

container._chartInstance = new Chart(canvas, {
type: chartType || "bar",
data: {
labels,
datasets: [{
label: "Value",
data: values,
backgroundColor: labels.map((_, idx) => palette[idx % palette.length]),
borderColor: labels.map((_, idx) => palette[idx % palette.length]),
borderWidth: 2,
fill: false,
tension: 0.25
}]
},
options: {
responsive: true,
maintainAspectRatio: true,
plugins: {
legend: {
display: chartType === "pie"
}
},
scales: chartType === "pie" ? {} : {
x: {
ticks: {
maxRotation: 30,
minRotation: 30
}
},
y: {
beginAtZero: true
}
}
}
})
}

function appendChartMessage(msg, chartData, chartType) {
if (!Array.isArray(chartData) || !chartData.length) {
return
}

const body = msg.querySelector(".message-body")
if (!body) {
return
}

let container = msg.querySelector(".chart-container")
if (!container) {
container = document.createElement("div")
container.className = "chart-container"
body.appendChild(container)
}

const serializedData = JSON.stringify(chartData)
if (msg.dataset.chartData === serializedData && msg.dataset.chartType === String(chartType || "bar")) {
return
}

msg.dataset.chartData = serializedData
msg.dataset.chartType = String(chartType || "bar")
renderChartCanvas(container, chartData, chartType)
scrollChatToBottom()
serializeChat()
}

function serializeChat() {
const messages = Array.from(chatBox().querySelectorAll(".message")).map(msg => {
const body = msg.querySelector(".message-body")
const confidence = msg.dataset.explicitConfidence
const sources = Array.from(msg.querySelectorAll(".source-item")).map(node => node.textContent || "")
return {
type: msg.classList.contains("user") ? "user" : "assistant",
text: body ? body.textContent : "",
html: body ? body.innerHTML : "",
queryId: msg.dataset.queryId || "",
confidence: confidence === undefined ? null : Number(confidence),
sources,
chartData: (() => {
try {
return JSON.parse(msg.dataset.chartData || "null")
} catch {
return null
}
})(),
chartType: msg.dataset.chartType || null
}
})
localStorage.setItem(STORAGE_KEYS.chat, JSON.stringify(messages))
}

function persistQueryHistory() {
localStorage.setItem(STORAGE_KEYS.history, JSON.stringify(queryHistory))
}

function clearStoredClientState() {
queryHistory = []
chatBox().innerHTML = ""
localStorage.removeItem(STORAGE_KEYS.chat)
localStorage.removeItem(STORAGE_KEYS.history)
updateHistory()
}

function restoreQueryHistory() {
try {
const saved = JSON.parse(localStorage.getItem(STORAGE_KEYS.history) || "[]")
if (Array.isArray(saved)) {
queryHistory = saved.filter(item => item && item.id && item.text)
}
} catch {
queryHistory = []
}
updateHistory()
}

function restoreChat() {
try {
const saved = JSON.parse(localStorage.getItem(STORAGE_KEYS.chat) || "[]")
if (!Array.isArray(saved) || !saved.length) {
return
}
chatBox().innerHTML = ""
saved.forEach(item => {
const msg = createMessage(item.type === "user" ? "user" : "assistant", item.html || item.text || "", true)
if (item.queryId) {
msg.dataset.queryId = item.queryId
}
if (item.confidence !== null && item.confidence !== undefined && !Number.isNaN(Number(item.confidence))) {
setMessageConfidence(msg, Number(item.confidence))
}
if (Array.isArray(item.sources) && item.sources.length) {
setMessageSources(msg, [], item.sources)
}
if (Array.isArray(item.chartData) && item.chartData.length) {
appendChartMessage(msg, item.chartData, item.chartType || "bar")
}
})
scrollChatToBottom()
} catch {
chatBox().innerHTML = ""
}
}

function escapeHtml(text) {
return String(text ?? "")
.replace(/&/g, "&amp;")
.replace(/</g, "&lt;")
.replace(/>/g, "&gt;")
}

function extractConfidenceText(text) {
const raw = String(text || "")
const match = raw.match(/\n*\s*Confidence\s+\(part\)\s*:\s*(\d+(?:\.\d+)?)%\s*$/i)
if (!match) {
return {
body: raw.replace(/\n*\s*Confidence\s*:\s*\d+(?:\.\d+)?%\s*$/i, "").trimEnd(),
confidence: null
}
}
return {
body: raw
.slice(0, match.index)
.replace(/\n*\s*Confidence\s*:\s*\d+(?:\.\d+)?%\s*$/i, "")
.trimEnd(),
confidence: Number(match[1])
}
}

function renderMarkdownTable(text) {
const lines = text.split("\n").map(line => line.trim()).filter(Boolean)
if (lines.length < 3) {
return ""
}

const rows = lines
.filter(line => line.startsWith("|") && line.endsWith("|"))
.map(line => line.split("|").slice(1, -1).map(cell => escapeHtml(cell.trim())))

if (rows.length < 2) {
return ""
}

const header = rows[0]
const bodyRows = rows.slice(2)
if (!bodyRows.length) {
return ""
}

let html = "<table><thead><tr>"
header.forEach(cell => {
html += `<th>${cell}</th>`
})
html += "</tr></thead><tbody>"
bodyRows.forEach(row => {
html += "<tr>"
header.forEach((_, idx) => {
html += `<td>${row[idx] || ""}</td>`
})
html += "</tr>"
})
html += "</tbody></table>"
return html
}

function isMarkdownTableLine(line) {
const value = String(line || "").trim()
return value.startsWith("|") && value.endsWith("|")
}

function isMarkdownDividerLine(line) {
const value = String(line || "").trim()
if (!isMarkdownTableLine(value)) {
return false
}
return value
.split("|")
.slice(1, -1)
.every(cell => /^:?-{3,}:?$/.test(cell.trim()))
}

function formatPlainBlocks(text) {
const normalized = String(text || "").replace(/\r\n/g, "\n").trim()
if (!normalized) {
return ""
}

const structured = normalized
    .replace(/(Question\s+\d+\s*:.*?)(?=\nQuestion\s+\d+\s*:|\Z)/gs, "\n\n$1\n\n")
    .replace(/(File\s+\d+\b.*?)(?=\nFile\s+\d+\b|\Z)/gs, "\n\n$1\n\n")
    .replace(/Conflict detected across files\./g, "Conflict detected across files.\n")

const blocks = structured.split(/\n{2,}/).filter(Boolean)
return blocks.map(block => {
const lines = block.split("\n").map(line => line.trim()).filter(Boolean)
if (!lines.length) {
return ""
}

const splitSemicolonList = value => {
if (!value.includes(";")) {
return null
}
const items = value.split(";").map(item => item.trim()).filter(Boolean)
return items.length >= 2 ? items : null
}

if (lines.every(line => /^[-*]\s+/.test(line))) {
return "<ul>" + lines.map(line => `<li>${escapeHtml(line.replace(/^[-*]\s+/, ""))}</li>`).join("") + "</ul>"
}

if (lines.every(line => /^\d+\.\s+/.test(line))) {
return "<ol>" + lines.map(line => `<li>${escapeHtml(line.replace(/^\d+\.\s+/, ""))}</li>`).join("") + "</ol>"
}

if (lines.length > 1 && lines.some(line => /^[-*]\s+/.test(line))) {
const firstPara = `<p>${escapeHtml(lines[0])}</p>`
const bulletLines = lines.slice(1).filter(line => /^[-*]\s+/.test(line))
if (bulletLines.length) {
return firstPara + "<ul>" + bulletLines.map(line => `<li>${escapeHtml(line.replace(/^[-*]\s+/, ""))}</li>`).join("") + "</ul>"
}
return firstPara
}

if (lines.length === 1) {
const parts = splitSemicolonList(lines[0])
if (parts) {
const intro = parts[0].endsWith(":") ? parts[0] : null
const items = intro ? parts.slice(1) : parts
if (intro && items.length) {
return `<p>${escapeHtml(intro)}</p><ul>` + items.map(item => `<li>${escapeHtml(item.replace(/^[\-\*\d]+\.\s*/, ""))}</li>`).join("") + "</ul>"
}
if (items.length >= 3) {
return "<ul>" + items.map(item => `<li>${escapeHtml(item.replace(/^[\-\*\d]+\.\s*/, ""))}</li>`).join("") + "</ul>"
}
}
}

if (lines.length > 1 && lines[0].endsWith(":")) {
const intro = `<p>${escapeHtml(lines[0])}</p>`
const remaining = lines.slice(1)
if (remaining.every(line => !line.endsWith(".") || /^\d+\.\s+/.test(line))) {
return intro + "<ul>" + remaining.map(line => `<li>${escapeHtml(line.replace(/^[\-\*\d]+\.\s*/, ""))}</li>`).join("") + "</ul>"
}
}

return "<p>" + lines.map(escapeHtml).join("<br>") + "</p>"
}).join("")
}

function renderMessage(msg, text) {
const body = msg.querySelector(".message-body")
if (!body) {
return
}
const parsed = extractConfidenceText(text)
body.innerHTML = formatResponse(parsed.body)
if (parsed.confidence !== null && parsed.confidence !== undefined) {
setMessageConfidence(msg, parsed.confidence)
} else if (!msg.dataset.explicitConfidence) {
setMessageConfidence(msg, null)
}
scrollChatToBottom()
}

function setMessageText(msg, text) {
renderMessage(msg, text)
serializeChat()
}

function appendMessageText(msg, text) {
    const body = msg.querySelector(".message-body")
    if (body) {
        body.textContent += text
        clearTimeout(msg._renderTimer)
        msg._renderTimer = setTimeout(() => {
            renderMessage(msg, body.textContent)
            serializeChat()
        }, 120)
    }
    scrollChatToBottom()
}

function setMessageConfidence(msg, confidence) {
let conf = msg.querySelector(".confidence")
if (conf) {
conf.remove()
}

if (confidence === null || confidence === undefined) {
delete msg.dataset.explicitConfidence
serializeChat()
return
}

msg.dataset.explicitConfidence = String(confidence)
conf = document.createElement("div")
conf.className = "confidence"
conf.textContent = "Confidence: " + confidence + "%"
msg.appendChild(conf)
serializeChat()
}

function setMessageSources(msg, chunks = [], sources = []) {
let box = msg.querySelector(".sources")
if (box) {
box.remove()
}

const labels = []
const seen = new Set()

if (Array.isArray(chunks)) {
chunks.forEach(chunk => {
const file = chunk && chunk.file ? String(chunk.file) : "Unknown source"
const page = chunk ? chunk.page : null
const label = page === null || page === undefined || page === "" || String(page) === "-1"
? file
: `${file} p.${page}`
if (!seen.has(label)) {
seen.add(label)
labels.push(label)
}
})
}

if (!labels.length && Array.isArray(sources)) {
sources.forEach(source => {
const label = String(source || "").trim()
if (label && !seen.has(label)) {
seen.add(label)
labels.push(label)
}
})
}

if (!labels.length) {
return
}

box = document.createElement("div")
box.className = "sources"

labels.forEach(label => {
const item = document.createElement("div")
item.className = "source-item"
item.textContent = label
box.appendChild(item)
})

msg.appendChild(box)
scrollChatToBottom()
serializeChat()
}

function closeHistoryMenus() {
document.querySelectorAll(".history-menu.open").forEach(menu => {
menu.classList.remove("open")
})
}

function deleteHistoryItem(id) {
queryHistory = queryHistory.filter(item => item.id !== id)
updateHistory()
}

function clearHistory() {
queryHistory = []
updateHistory()
}

function clearChat() {
const queryIdsInChat = new Set(
Array.from(chatBox().querySelectorAll(".message.user[data-query-id]"))
.map(node => node.dataset.queryId)
.filter(Boolean)
)

if (queryIdsInChat.size) {
queryHistory = queryHistory.filter(item => !queryIdsInChat.has(item.id))
}

chatBox().innerHTML = ""
document.getElementById("query").value = ""
closeHistoryMenus()
updateHistory()
serializeChat()
}

function confirmClearChat() {
const hasMessages = chatBox().children.length > 0
if (!hasMessages) {
return
}

const confirmed = window.confirm(
"Clear the current chat and remove the matching queries from Query History?"
)

if (!confirmed) {
return
}

const doubleConfirmed = window.confirm(
"This will permanently delete the visible chat and its related query history entries. Continue?"
)

if (!doubleConfirmed) {
return
}

clearChat()
}

function updateHistory() {
const box = document.getElementById("history")
const clearButton = document.getElementById("clear-history")
box.innerHTML = ""
persistQueryHistory()

if (clearButton) {
clearButton.disabled = queryHistory.length === 0
}

if (!queryHistory.length) {
box.textContent = "No queries yet."
return
}

queryHistory.slice().reverse().forEach(item => {
const row = document.createElement("div")
row.className = "history-item"

const label = document.createElement("button")
label.type = "button"
label.className = "history-label"
label.textContent = item.text
label.onclick = () => {
document.getElementById("query").value = item.text
}

const actions = document.createElement("div")
actions.className = "history-actions"

const menuButton = document.createElement("button")
menuButton.type = "button"
menuButton.className = "history-menu-trigger"
menuButton.textContent = "..."
menuButton.onclick = event => {
event.stopPropagation()
const menu = actions.querySelector(".history-menu")
const isOpen = menu.classList.contains("open")
closeHistoryMenus()
if (!isOpen) {
menu.classList.add("open")
}
}

const menu = document.createElement("div")
menu.className = "history-menu"

const deleteButton = document.createElement("button")
deleteButton.type = "button"
deleteButton.className = "history-delete-button"
deleteButton.textContent = "Delete"
deleteButton.onclick = event => {
event.stopPropagation()
deleteHistoryItem(item.id)
}

menu.appendChild(deleteButton)
actions.appendChild(menuButton)
actions.appendChild(menu)
row.appendChild(label)
row.appendChild(actions)
box.appendChild(row)
})
}

function renderFiles(files) {
const box = document.getElementById("files-list")
const clearFilesButton = document.getElementById("clear-files")
const sessionNote = document.getElementById("file-session-note")
box.innerHTML = ""

if (clearFilesButton) {
clearFilesButton.disabled = !Array.isArray(files) || files.length === 0
}

if (sessionNote) {
if (Array.isArray(files) && files.length) {
sessionNote.textContent = `${files.length} file${files.length === 1 ? "" : "s"} already uploaded in this session. Use the picker only to add more files.`
} else {
sessionNote.textContent = "No files uploaded yet."
}
}

if (!Array.isArray(files) || !files.length) {
box.textContent = "No uploaded files."
return
}

files.forEach(file => {
const fileName = typeof file === "string" ? file : (file && file.name) || "Unknown file"
const pageCount = typeof file === "string" ? null : file && file.pages
const div = document.createElement("div")
div.className = "file-item"

const header = document.createElement("div")
header.className = "file-header"

const label = document.createElement("span")
label.textContent = fileName
label.className = "file-name"

header.appendChild(label)

if (pageCount !== null && pageCount !== undefined) {
const meta = document.createElement("span")
meta.className = "file-pages"
meta.textContent = `${pageCount} page${pageCount === 1 ? "" : "s"}`
header.appendChild(meta)
}

const actions = document.createElement("div")
actions.className = "file-actions"

const btn = document.createElement("button")
btn.type = "button"
btn.textContent = "Delete"
btn.onclick = () => deleteFile(fileName, div)

div.appendChild(header)
actions.appendChild(btn)
div.appendChild(actions)
box.appendChild(div)
})
}

function loadFiles() {
return fetch("/files")
.then(res => res.json())
.then(data => {
 const files = Array.isArray(data.files) ? data.files : []
 renderFiles(files)
 if (!data.session_id || files.length === 0) {
 clearStoredClientState()
 }
 return data
})
}

function resetConversationUi() {
clearChat()
}

function resetUploadUi() {
document.getElementById("progress-bar").style.width = "0%"
document.getElementById("files").value = ""
}

function setStatusMessage(message, clearAfterMs = 0) {
const statusEl = document.getElementById("status")
statusEl.textContent = message || ""
if (statusClearTimer) {
clearTimeout(statusClearTimer)
statusClearTimer = null
}
if (clearAfterMs > 0) {
statusClearTimer = setTimeout(() => {
statusEl.textContent = ""
statusClearTimer = null
}, clearAfterMs)
}
}

function cancelUploadPolling() {
uploadPollRun += 1
}

function removeFileRow(fileRow) {
if (!fileRow) {
return
}

const list = document.getElementById("files-list")
fileRow.remove()
if (!list.querySelector(".file-item")) {
list.textContent = "No uploaded files."
}
}

function deleteFile(name, fileRow) {
cancelUploadPolling()
removeFileRow(fileRow)
fetch("/delete_file", {
method: "POST",
headers: { "Content-Type": "application/json" },
body: JSON.stringify({ filename: name })
})
.then(async res => {
const data = await res.json().catch(() => ({}))
if (!res.ok) {
throw new Error(data.error || "Delete failed.")
}
resetConversationUi()
resetUploadUi()
return loadFiles().then(() => data)
})
.then(data => {
setStatusMessage(data.message || "File deleted.", 3000)
})
.catch(err => {
loadFiles()
setStatusMessage(err.message || "Delete failed.")
})
}

function deleteAllFiles() {
const list = document.getElementById("files-list")
if (!list.querySelector(".file-item")) {
return
}

const confirmed = window.confirm("Delete all uploaded files for the current session?")
if (!confirmed) {
return
}

const doubleConfirmed = window.confirm("This will permanently remove all uploaded files and their indexed data. Continue?")
if (!doubleConfirmed) {
return
}

cancelUploadPolling()
fetch("/delete_all_files", {
method: "POST",
headers: { "Content-Type": "application/json" }
})
.then(async res => {
const data = await res.json().catch(() => ({}))
if (!res.ok) {
throw new Error(data.error || "Delete all failed.")
}
resetConversationUi()
resetUploadUi()
return loadFiles().then(() => data)
})
.then(data => {
setStatusMessage(data.message || "All files deleted.", 3000)
})
.catch(err => {
loadFiles()
setStatusMessage(err.message || "Delete all failed.")
})
}

function wait(ms) {
return new Promise(resolve => setTimeout(resolve, ms))
}

async function pollUploadStatus(runId) {
const statusEl = document.getElementById("status")
const progressBar = document.getElementById("progress-bar")
const fallbackProgressMap = {
queued: 10,
processing: 25,
embedding: 70,
indexing: 85,
ready: 100,
completed: 100,
failed: 100,
idle: 0
}

const startedAt = Date.now()
const maxPollMs = 30 * 60 * 1000

while (Date.now() - startedAt < maxPollMs) {
if (runId !== uploadPollRun) {
return
}

const res = await fetch("/upload_status")
const data = await res.json()

if (runId !== uploadPollRun) {
return
}

const state = data.state || data.status || "idle"
const filesTotal = Number(data.files_total || 0)
const filesProcessed = Number(data.files_processed || 0)
const chunksPrepared = Number(data.chunks_prepared || 0)
const chunksIndexed = Number(data.chunks_indexed || 0)

if (state === "idle") {
return
}

let progress = fallbackProgressMap[state] || 0
if (state === "processing" && filesTotal > 0) {
progress = 15 + Math.round((filesProcessed / filesTotal) * 45)
} else if (state === "embedding" && chunksPrepared > 0) {
progress = 60 + Math.round((chunksIndexed / chunksPrepared) * 10)
} else if (state === "indexing" && chunksPrepared > 0) {
progress = 75 + Math.round((chunksIndexed / chunksPrepared) * 24)
}

const progressParts = []
if (filesTotal > 0) {
progressParts.push(`${filesProcessed}/${filesTotal} files`)
}
if (state === "indexing" && chunksPrepared > 0) {
progressParts.push(`${chunksIndexed}/${chunksPrepared} vectors`)
}

const detail = progressParts.length ? ` (${progressParts.join(", ")})` : ""
statusEl.textContent = (data.message || state) + detail
progressBar.style.width = Math.min(progress, 100) + "%"

if (state === "ready" || state === "completed") {
loadFiles()
return
}

if (state === "failed") {
return
}

await wait(1500)
}

statusEl.textContent = "Upload is still running. Progress polling timed out."
}

function uploadFiles() {
const input = document.getElementById("files")
const files = input.files
const statusEl = document.getElementById("status")
const progressBar = document.getElementById("progress-bar")

if (!files.length) {
statusEl.textContent = "Select files."
return
}

const form = new FormData()
for (const file of files) {
form.append("files", file)
}

resetConversationUi()
statusEl.textContent = "Uploading files..."
progressBar.style.width = "10%"
const runId = ++uploadPollRun

fetch("/upload", {
method: "POST",
body: form
})
.then(async res => {
const data = await res.json()
statusEl.textContent = data.message || "Upload started."
progressBar.style.width = "20%"
loadFiles()
if (!res.ok) {
throw new Error(data.error || data.message || "Upload failed.")
}
return pollUploadStatus(runId)
})
.catch(err => {
statusEl.textContent = err.message || "Upload failed."
progressBar.style.width = "0%"
})
}


function formatResponse(text) {
    if (!text) return "";
    const normalized = String(text).replace(/\r\n/g, "\n").trim()
    if (!normalized) return ""

    const lines = normalized.split("\n")
    const parts = []
    let plainBuffer = []

    const flushPlainBuffer = () => {
        const plainText = plainBuffer.join("\n").trim()
        if (plainText) {
            parts.push(formatPlainBlocks(plainText))
        }
        plainBuffer = []
    }

    for (let i = 0; i < lines.length; i++) {
        const current = lines[i]
        const next = lines[i + 1]

        if (isMarkdownTableLine(current) && isMarkdownDividerLine(next)) {
            flushPlainBuffer()
            const tableLines = [current, next]
            i += 2
            while (i < lines.length && isMarkdownTableLine(lines[i])) {
                tableLines.push(lines[i])
                i += 1
            }
            i -= 1
            const tableHtml = renderMarkdownTable(tableLines.join("\n"))
            if (tableHtml) {
                parts.push(tableHtml)
                continue
            }
            plainBuffer.push(...tableLines)
            continue
        }

        plainBuffer.push(current)
    }

    flushPlainBuffer()
    return parts.join("")
}

function processStreamLine(line, msg) {
if (!line.trim()) return

let data
try {
data = JSON.parse(line)
} catch {
appendMessageText(msg, line)
return
}

if (data.type === "meta") {
setMessageConfidence(msg, data.confidence)
setMessageSources(msg, data.chunks, data.sources)
return
}

if (data.type === "token") {
appendMessageText(msg, data.token || "")
return
}

if (data.type === "chart") {
appendChartMessage(msg, data.data, data.chart_type)
return
}
}

async function ask() {
const input = document.getElementById("query")
const q = input.value.trim()
if (!q) return
 
const queryItem = { id: crypto.randomUUID(), text: q }
createUserMessage(queryItem)
queryHistory.push(queryItem)
updateHistory()
input.value = ""

const msg = createMessage(
"assistant",
'Thinking <span class="dot">.</span><span class="dot">.</span><span class="dot">.</span>',
true
)
serializeChat()

try {
const response = await fetch("/query_stream", {
method: "POST",
headers: { "Content-Type": "application/json" },
body: JSON.stringify({ query: q })
})

if (!response.ok || !response.body) {
const text = await response.text()
setMessageText(msg, text || "Request failed.")
return
}

setMessageText(msg, "")

const reader = response.body.getReader()
const decoder = new TextDecoder()
let buffer = ""

while (true) {
const { done, value } = await reader.read()
if (done) {
buffer += decoder.decode()
break
}

buffer += decoder.decode(value, { stream: true })
const lines = buffer.split("\n")
buffer = lines.pop()
lines.forEach(line => processStreamLine(line, msg))
}

if (buffer.trim()) {
    processStreamLine(buffer, msg)
}

const body = msg.querySelector(".message-body")
if (body && !body.textContent.trim()) {
body.textContent = "No answer returned."
if (body) {
    body.innerHTML = formatResponse(body.textContent)
}
}
} catch (err) {
setMessageText(msg, err.message || "Unable to reach backend.")
}
}

document.addEventListener("click", event => {
if (!event.target.closest(".history-actions")) {
closeHistoryMenus()
}
})

const queryInput = document.getElementById("query")
queryInput.addEventListener("keydown", event => {
if (event.key === "Enter" && !event.shiftKey) {
event.preventDefault()
ask()
}
})

const clearHistoryButton = document.getElementById("clear-history")
if (clearHistoryButton) {
clearHistoryButton.onclick = () => {
clearHistory()
}
}

const clearChatButton = document.getElementById("clear-chat")
if (clearChatButton) {
clearChatButton.onclick = () => {
confirmClearChat()
}
}

const clearFilesButton = document.getElementById("clear-files")
if (clearFilesButton) {
clearFilesButton.onclick = () => {
deleteAllFiles()
}
}

restoreQueryHistory()
restoreChat()
loadFiles()

<script lang="ts">
  import { onMount, tick } from 'svelte';
  import MarkdownContent from '$lib/components/app/content/MarkdownContent/MarkdownContent.svelte';
  import Button from '$lib/components/ui/button/button.svelte';
  import Textarea from '$lib/components/ui/textarea/textarea.svelte';
  import { DatabaseService as DB } from '$lib/services/database.service';
  import type { DatabaseMessage, DatabaseConversation, DatabaseMessageExtraImageFile } from '$lib/types/database';
  import { AttachmentType } from '$lib/enums';
  import { settingsStore } from '$lib/stores/settings/index.svelte';
  import { streamChat } from './api';

  type Message = DatabaseMessage & { uiStatus?: string };
  let props = $state<any>(null);
  let conversations = $state<DatabaseConversation[]>([]);
  let active = $state<DatabaseConversation | null>(null);
  let messages = $state<Message[]>([]);
  let input = $state('');
  let images = $state<{name: string; base64Url: string}[]>([]);
  let busy = $state(false);
  let loadingHistory = $state(false);
  let stopping = $state(false);
  let error = $state('');
  let settingsOpen = $state(false);
  let controller: AbortController | null = null;
  let bottom: HTMLDivElement;
  let thinking = $state('default');
  let effort = $state('');
  let budget = $state('');
  let output = $state('');
  let sampling = $state<Record<string, string>>({temperature:'', top_p:'', top_k:'', min_p:''});
  const keys = ['temperature', 'top_p', 'top_k', 'min_p'];
  let effectiveThinking = $derived(thinking === 'default' ? props?.kvmem.defaults.enable_thinking : thinking === 'on');
  let defaults = $derived(props?.kvmem.sampling[effectiveThinking ? 'thinking' : 'non_thinking'] ?? {});

  onMount(() => {
    settingsStore.initialize();
    void Promise.all([loadProps(), refreshHistory()]);
    return () => controller?.abort();
  });
  async function loadProps() {
    try {
      const res = await fetch('./props');
      if (!res.ok) throw new Error(`连接失败 (${res.status})`);
      props = await res.json();
    } catch (e) { error = String(e); }
  }
  async function refreshHistory() {
    try { conversations = await DB.getAllConversations(); }
    catch (e) { error = `无法保存浏览器历史：${String(e)}`; }
  }
  async function selectConversation(conversation: DatabaseConversation | null) {
    if (busy || loadingHistory) return;
    loadingHistory=true;
    try {
      const loaded = conversation ? (await DB.getConversationMessages(conversation.id)).map(m => {
        const message = m as Message;
        return {...message, uiStatus: message.uiStatus === 'pending' ? 'interrupted' : message.uiStatus};
      }) : [];
      active=conversation; messages=loaded;
      images=[]; input=''; error='';
      await scrollToBottom();
    } catch(e) { error=`读取历史失败：${String(e)}`; }
    finally { loadingHistory=false; }
  }
  async function removeConversation() {
    if (!active || busy || loadingHistory) return;
    try { await DB.deleteConversation(active.id); await selectConversation(null); await refreshHistory(); }
    catch (e) { error = String(e); }
  }
  async function scrollToBottom() { await tick(); bottom?.scrollIntoView({block:'end'}); }
  async function addImages(files: File[]) {
    if (busy || !props?.modalities.vision) return;
    try {
      for (const file of files) {
        if (!['image/png','image/jpeg','image/webp','image/gif'].includes(file.type)) throw new Error('支持 PNG、JPEG、WebP 和 GIF 图片');
        if (file.size > 10 * 1024 * 1024 || images.length >= 4) throw new Error('每张图片不超过 10 MiB，每条最多 4 张');
        const url = await new Promise<string>((resolve,reject) => {
          const reader = new FileReader(); reader.onload=()=>resolve(String(reader.result)); reader.onerror=reject; reader.readAsDataURL(file);
        });
        images = [...images, {name:file.name, base64Url:url}];
      }
    } catch (e) { error = String(e); }
  }
  function wire(message: Message) {
    const pictures = (message.extra ?? []).filter((x): x is DatabaseMessageExtraImageFile => x.type === AttachmentType.IMAGE);
    return {
      role: message.role,
      content: pictures.length ? [{type:'text',text:message.content}, ...pictures.map(x => ({type:'image_url',image_url:{url:x.base64Url}}))] : message.content,
      ...(message.reasoningContent ? {reasoning_content:message.reasoningContent} : {})
    };
  }
  function requestOptions() {
    const body: Record<string, unknown> = {stream:true, model:props.model_name};
    if (thinking !== 'default') body.chat_template_kwargs = {enable_thinking:thinking === 'on'};
    if (effort.trim()) body.reasoning_effort = effort.trim();
    if (budget !== '') {
      const n=Number(budget); if (!Number.isInteger(n) || n<0) throw new Error('思考预算必须是非负整数');
      body.reasoning_budget_tokens=n;
    }
    if (output !== '') {
      const n=Number(output); if (!Number.isInteger(n) || n<1 || n>props.kvmem.generation_limit) throw new Error(`最大输出必须在 1..${props.kvmem.generation_limit} 内`);
      body.max_tokens=n;
    }
    for (const key of keys) if (sampling[key] !== '') {
      const n=Number(sampling[key]); if (!Number.isFinite(n)) throw new Error(`${key} 不是有效数字`);
      body[key]=n;
    }
    return body;
  }
  async function send() {
    if (busy || loadingHistory || !props || (!input.trim() && !images.length)) return;
    error='';
    let options;
    try { options=requestOptions(); } catch(e) { error=String(e); return; }
    busy=true; stopping=false; controller=new AbortController();
    let reply: Message | undefined;
    try {
      const conversation: DatabaseConversation = active ?? await DB.createConversation(input.trim().slice(0,40) || '图片对话');
      active=conversation;
      const user=await DB.createMessageBranch({convId:conversation.id, type:'text', role:'user', content:input,
        timestamp:Date.now(), children:[], parent:null, extra:images.map(x=>({...x,type:AttachmentType.IMAGE}))}, messages.at(-1)?.id ?? null);
      messages=[...messages,user]; input=''; images=[];
      const history=messages.filter(m=>m.content || m.reasoningContent || m.extra?.length).map(wire);
      const pending = {convId:conversation.id, type:'text' as const, role:'assistant' as const, content:'', reasoningContent:'',
        timestamp:Date.now()+1, children:[], parent:user.id, uiStatus:'pending'};
      const assistant: Message=await DB.createMessageBranch(pending,user.id);
      reply=assistant;
      messages=[...messages,assistant]; await scrollToBottom();
      await streamChat({...options,messages:history},controller.signal,data=>{
        const delta=data.choices?.[0]?.delta;
        if (delta?.tool_calls) throw new Error('此聊天页不执行工具，请使用支持工具的客户端。');
        reply!.content+=delta?.content ?? '';
        reply!.reasoningContent=(reply!.reasoningContent ?? '')+(delta?.reasoning_content ?? '');
        if (data.timings) reply!.timings=data.timings;
        messages=messages.map(m=>m.id===reply!.id ? {...reply!} : m);
        if (bottom && bottom.getBoundingClientRect().top < window.innerHeight+220) void scrollToBottom();
      });
      assistant.uiStatus='complete';
    } catch(e) {
      if (reply) reply.uiStatus=controller?.signal.aborted ? 'interrupted' : 'error';
      if (!controller?.signal.aborted) error=String(e);
    } finally {
      try {
        if (reply) {
          messages=messages.map(m=>m.id===reply!.id ? {...reply!} : m);
          const {id,...saved}=reply; await DB.updateMessage(id,saved);
        }
        if (active) await DB.updateConversation(active.id,{lastModified:Date.now()});
        await refreshHistory();
      } catch(e) { error=`保存历史失败：${String(e)}`; }
      busy=false; stopping=false; controller=null;
    }
  }
  function stop() { stopping=true; controller?.abort(); }
  async function copy(text: string) { try { await navigator.clipboard.writeText(text); } catch(e) { error=String(e); } }
  function download(text: string) {
    const url=URL.createObjectURL(new Blob([text],{type:'text/markdown;charset=utf-8'}));
    const link=document.createElement('a'); link.href=url; link.download='answer.md'; link.click(); setTimeout(()=>URL.revokeObjectURL(url),1000);
  }
</script>

<svelte:head><title>KVMem Chat</title><meta name="description" content="Local text and image chat with KVMem" /></svelte:head>
<div class="shell">
  <aside aria-label="聊天历史">
    <h1>KVMem <span>Chat</span></h1>
    <Button onclick={()=>selectConversation(null)} disabled={busy || loadingHistory}>新建对话</Button>
    <nav>{#each conversations as conversation (conversation.id)}
      <button class:selected={conversation.id===active?.id} disabled={busy || loadingHistory} onclick={()=>selectConversation(conversation)}>{conversation.name}</button>
    {/each}</nav>
    <small>历史仅保存在此浏览器</small>
  </aside>
  <main>
    <header>
      <div><strong>{props?.model_name ?? '正在连接...'}</strong><small>{props ? `${props.default_generation_settings.n_ctx.toLocaleString()} 上下文 · ${props.modalities.vision ? '支持图片' : '文本'}` : ''}</small></div>
      <div class="actions"><Button variant="outline" onclick={()=>settingsOpen=!settingsOpen}>参数</Button>{#if active}<Button variant="ghost" disabled={busy || loadingHistory} onclick={removeConversation}>删除对话</Button>{/if}</div>
    </header>
    {#if settingsOpen}
      <section class="settings" aria-label="生成参数">
        <p>留空即跟随服务器。思考强度与思考 token 预算分别设置。</p>
        <label>思考<select bind:value={thinking} disabled={busy || loadingHistory}><option value="default">跟随服务器（{props?.kvmem.defaults.enable_thinking ? '开启' : '关闭'}）</option><option value="on">开启</option><option value="off">关闭</option></select></label>
        <label>模型思考强度<input aria-label="模型思考强度" bind:value={effort} disabled={busy || loadingHistory} placeholder={props?.kvmem.defaults.chat_template_kwargs.reasoning_effort ?? '跟随模板'} list="efforts" /><datalist id="efforts"><option value="low"></option><option value="medium"></option><option value="xhigh"></option></datalist></label>
        <label>思考 token 预算<input aria-label="思考 token 预算" type="text" inputmode="numeric" bind:value={budget} disabled={busy || loadingHistory} placeholder={String(props?.kvmem.defaults.reasoning_budget_tokens ?? '')} /></label>
        <label>最大输出（含思考）<input aria-label="最大输出" type="text" inputmode="numeric" bind:value={output} disabled={busy || loadingHistory} placeholder={String(props?.default_generation_settings.params.max_tokens ?? '')} /></label>
        {#each keys as key}<label>{key}<input aria-label={key} bind:value={sampling[key]} disabled={busy || loadingHistory} inputmode="decimal" placeholder={defaults[key] === undefined ? '' : String(Number(Number(defaults[key]).toPrecision(6)))} /></label>{/each}
      </section>
    {/if}
    <section class="messages" aria-label="消息">
      {#if !messages.length}<div class="welcome"><h2>有什么想一起完成？</h2><p>问一个问题，或让模型写一段网页代码。</p></div>{/if}
      {#each messages as message (message.id)}
        <article class:user={message.role==='user'}>
          <strong class="role">{message.role==='user' ? '你' : 'KVMem'}</strong>
          {#each message.extra ?? [] as attachment}{#if attachment.type===AttachmentType.IMAGE}<img class="attachment" src={attachment.base64Url} alt={attachment.name} />{/if}{/each}
          {#if message.reasoningContent}<details><summary>思考</summary><div class="reasoning">{message.reasoningContent}</div></details>{/if}
          {#if message.role==='assistant'}<MarkdownContent content={message.content} />{:else}<div class="user-text">{message.content}</div>{/if}
          {#if message.role==='assistant'}
            <footer>
              {#if message.uiStatus==='pending'}<span>{stopping ? '正在停止...' : '正在生成...'}</span>{/if}
              {#if message.uiStatus==='interrupted'}<span>已中止</span>{/if}
              {#if message.uiStatus==='error'}<span>生成失败</span>{/if}
              {#if message.timings?.predicted_ms && message.timings.predicted_n !== undefined}<span>{(1000*message.timings.predicted_n/message.timings.predicted_ms).toFixed(2)} tok/s · {message.timings.predicted_n} token</span>{/if}
              {#if message.content}<button onclick={()=>copy(message.content)}>复制</button><button onclick={()=>download(message.content)}>下载</button>{/if}
            </footer>
          {/if}
        </article>
      {/each}
      <div bind:this={bottom}></div>
    </section>
    <form onsubmit={event=>{event.preventDefault();void send();}}>
      {#if error}<p class="error" role="alert">{error}</p>{#if !props}<button type="button" onclick={loadProps}>重新连接</button>{/if}{/if}
      <div class="previews">{#each images as picture,i}<div><img src={picture.base64Url} alt={picture.name}/><button type="button" aria-label="移除图片" disabled={busy || loadingHistory} onclick={()=>images=images.filter((_,n)=>n!==i)}>×</button></div>{/each}</div>
      <Textarea aria-label="消息输入" bind:value={input} disabled={busy || loadingHistory || !props} placeholder="输入消息，Enter 发送，Shift+Enter 换行"
        onkeydown={event=>{if(event.key==='Enter' && !event.shiftKey && !event.isComposing){event.preventDefault();void send();}}}
        onpaste={event=>{const files=Array.from(event.clipboardData?.files ?? []);if(files.length && props?.modalities.vision){event.preventDefault();void addImages(files);}}} />
      <div class="compose-actions"><div>{#if props?.modalities.vision}<label class="upload">添加图片<input aria-label="添加图片" type="file" accept="image/png,image/jpeg,image/webp,image/gif" multiple disabled={busy || loadingHistory} onchange={event=>{void addImages(Array.from(event.currentTarget.files ?? []));event.currentTarget.value='';}} /></label>{/if}</div>
        {#if busy}<Button type="button" variant="outline" onclick={stop} disabled={stopping}>{stopping ? '正在停止...' : '停止生成'}</Button>{:else}<Button type="submit" disabled={loadingHistory || !props || (!input.trim() && !images.length)}>发送</Button>{/if}</div>
    </form>
  </main>
</div>

<style>
  :global(body){margin:0;background:#f7f8fa;color:#20252e;font-family:system-ui,sans-serif}
  .shell{display:grid;grid-template-columns:240px minmax(0,1fr);min-height:100dvh}
  aside{padding:24px 16px;border-right:1px solid #e3e6eb;background:#eef1f5;display:flex;flex-direction:column;gap:20px;position:sticky;top:0;height:100dvh}
  h1{font-size:21px;font-weight:700}h1 span{font-weight:400;color:#637184}nav{overflow:auto;flex:1}nav button{display:block;text-align:left;width:100%;padding:10px;border-radius:8px;margin-bottom:4px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}nav button:hover,.selected{background:#dde5ef}small{display:block;color:#6c7787;font-size:12px}
  main{width:100%;max-width:1040px;margin:0 auto;padding:0 32px}header{display:flex;justify-content:space-between;align-items:center;gap:10px;padding:20px 0;border-bottom:1px solid #e3e6eb}header strong{font-size:14px;overflow-wrap:anywhere}.actions{display:flex;gap:6px;flex-shrink:0}
  .messages{padding:16px 0}.welcome{text-align:center;padding:12vh 0;color:#647085}.welcome h2{font-size:27px;color:#283747;margin-bottom:10px}article{margin:12px 0 24px;padding:18px;border-radius:12px;background:white;overflow-wrap:anywhere}.user{background:#eaf0f7}.role{font-size:12px;color:#617085;display:block;margin-bottom:8px}.user-text,.reasoning{white-space:pre-wrap}.reasoning{color:#637184;font-size:14px;padding:12px 0}details{margin:8px 0}summary{cursor:pointer;color:#637184;font-size:13px}.attachment{max-width:320px;max-height:240px;border-radius:6px;margin:8px 0}
  footer{display:flex;gap:14px;align-items:center;flex-wrap:wrap;color:#748094;font-size:12px;margin-top:12px}footer button:hover{text-decoration:underline}form{position:sticky;bottom:0;background:#f7f8fa;padding:12px 0 18px;border-top:1px solid #e3e6eb}.compose-actions{display:flex;justify-content:space-between;align-items:center;margin-top:10px}.upload{font-size:13px;cursor:pointer}.upload input{display:none}.previews{display:flex;gap:8px}.previews>div{position:relative;margin-bottom:8px}.previews img{height:64px;border-radius:6px}.previews button{position:absolute;right:0;top:0;background:white;border-radius:50%;padding:0 5px}.error{color:#a12b2b;font-size:14px;padding-bottom:10px}
  .settings{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;padding:18px 0}.settings p{grid-column:1/-1;color:#657080;font-size:13px}.settings label{font-size:12px}.settings input,.settings select{width:100%;padding:7px;border:1px solid #d5dce5;border-radius:6px;background:white;margin-top:5px;font-size:13px}
  @media(max-width:720px){.shell{grid-template-columns:1fr}aside{position:static;height:auto;padding:12px;gap:10px}aside nav{display:flex;max-height:90px}aside nav button{max-width:160px;min-width:120px}aside small{display:none}main{padding:0 14px}.settings{grid-template-columns:repeat(2,minmax(0,1fr))}header{align-items:flex-start}article{padding:12px}.welcome{padding:5vh 0}}
</style>

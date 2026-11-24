/* Data tasks share the main conversation; task adapters come from the API. */
let dataChat = null;
let intakeHistory = [];
let previousChoices = null;
let activeTrainingSession = '';
let activeRequests = 0;
let restoringSession = false;
const SESSION_INDEX = 'trainpilot.training.sessions.v1';
const SESSION_PREFIX = 'trainpilot.training.session.';
const sessionIndex = () => {try {const value=JSON.parse(localStorage.getItem(SESSION_INDEX)||'[]');return Array.isArray(value)?value.filter(x=>x&&/^[a-f0-9]{32}$/.test(x.id)):[]} catch{return []}};
function reportURL(id){return '/api/training/'+encodeURIComponent(id)+'/report?session='+encodeURIComponent(activeTrainingSession)}
function serializeMessage(element){const copy=element.cloneNode(true);copy.querySelectorAll('details,.comparison-chart,a').forEach(x=>x.remove());const details=Array.from(element.querySelectorAll('details')).map(d=>({title:d.querySelector('summary')?.textContent||'Details',text:Array.from(d.querySelectorAll('p')).map(p=>p.textContent).join('\n')}));const link=element.querySelector('a[href*="/api/training/"]');const match=link?.getAttribute('href')?.match(/\/api\/training\/([a-f0-9]{32})\/report/);let attempts=[],ablation=null,discovery=null,comparison=null;try{attempts=JSON.parse(element.dataset.attempts||'[]');ablation=JSON.parse(element.dataset.ablation||'null');discovery=JSON.parse(element.dataset.discovery||'null');comparison=JSON.parse(element.dataset.comparison||'null')}catch{}return{className:element.className,text:copy.textContent,details,reportId:match?.[1],attempts,ablation,discovery,comparison}}
function rememberSession(){if(restoringSession||!activeTrainingSession)return;try{const info=dataChat?.info?{...dataChat.info}:null;if(info)delete info.examples;const state=dataChat?{...dataChat,info}:null;const messages=Array.from($('chat-messages').children).map(serializeMessage);const choice=!$('choice-popup').hidden?previousChoices:null;localStorage.setItem(SESSION_PREFIX+activeTrainingSession,JSON.stringify({dataChat:state,intakeHistory,previousChoices,choice,messages,draft:$('request-text').value}));}catch{}}
function showSession(id){const list=sessionIndex(),entry=list.find(x=>x.id===id);if(!entry)return;restoringSession=true;let state={};try{state=JSON.parse(localStorage.getItem(SESSION_PREFIX+id)||'{}')}catch{}activeTrainingSession=id;sessionStorage.setItem('trainpilot.activeSession',id);$('session-id').value=id;dataChat=state.dataChat||null;intakeHistory=Array.isArray(state.intakeHistory)?state.intakeHistory:[];previousChoices=state.previousChoices||null;$('chat-messages').replaceChildren();(state.messages||[]).forEach(m=>{const el=document.createElement('div');el.className=typeof m.className==='string'?m.className:'msg assistant';el.textContent=String(m.text||'');(m.details||[]).forEach(d=>{const details=document.createElement('details'),summary=document.createElement('summary'),content=document.createElement('p');summary.textContent=d.title;content.textContent=d.text;details.append(summary,content);el.append(details)});if(Array.isArray(m.attempts)&&m.attempts.length){el.dataset.attempts=JSON.stringify(m.attempts);modelTable(el,m.attempts);chart(el,'Model comparison (validation set)',m.attempts.map(a=>({name:a.model,value:a.validation_score})),m.attempts[0].metric==='mae'?'MAE, lower is better':'Macro F1, higher is better')}if(m.ablation){el.dataset.ablation=JSON.stringify(m.ablation);ablationTable(el,m.ablation)}if(m.discovery){el.dataset.discovery=JSON.stringify(m.discovery);discoveryCard(el,m.discovery)}if(m.comparison){versionChart(el,m.comparison)}if(m.reportId&&/^[a-f0-9]{32}$/.test(m.reportId)){const link=document.createElement('a');link.textContent='Open experiment report';link.href=reportURL(m.reportId);link.target='_blank';link.rel='noopener';link.style.cssText='display:block;margin-top:10px';el.append(link)}$('chat-messages').appendChild(el)});if(!state.messages?.length){const intro=document.createElement('div');intro.className='msg system';intro.textContent='Describe your training goal or upload a CSV using +. Each session keeps its data and conversation separate.';$('chat-messages').appendChild(intro)}$('request-text').value=state.draft||'';fileInput.value='';$('chat-file-name').textContent='Attach a CSV and describe your goal';$('training-popup').hidden=true;$('choice-popup').hidden=true;if(state.choice)showChoices(state.choice.message,state.choice.options);$('training-session-select').value=id;$('run-status').textContent=dataChat?.status==='completed'?'Training completed':'Ready';restoringSession=false;scrollChat()}
function refreshSessionSelect(){const select=$('training-session-select');select.replaceChildren();sessionIndex().forEach(item=>{const option=document.createElement('option');option.value=item.id;option.textContent=item.label;select.append(option)});select.value=activeTrainingSession}
function createTrainingSession(){const list=sessionIndex(),id=crypto.randomUUID().replaceAll('-','');list.push({id,label:'Training '+(list.length+1)});localStorage.setItem(SESSION_INDEX,JSON.stringify(list));refreshSessionSelect();showSession(id);refreshSessionSelect();rememberSession()}
function sessionBusy(){return $('btn-run').disabled||activeRequests>0||!!fileInput.files.length}
const petColors={data:['#6589cf','#d9e7ff'],model:['#d1789c','#ffe0eb'],review:['#6ca688','#daf5e5'],research:['#a785cb','#efe2ff'],version:['#d28d4c','#fff0d7']};
const activePets=new Map();
function showChoices(message,options){const box=$('choice-popup'),container=$('choice-options');container.replaceChildren();if(!Array.isArray(options)||!options.length){box.hidden=true;return} previousChoices={message,options};$('choice-question').textContent=message||'Please choose';options.slice(0,4).forEach(option=>{const button=document.createElement('button');button.type='button';button.textContent=option;button.addEventListener('click',()=>{box.hidden=true;$('request-text').value=option;$('btn-run').click()});container.append(button)});box.hidden=false}
function splitPet(role){if(activePets.has(role)||activePets.size>=5)return;const main=$('pet-main');const pet=main.cloneNode(true);pet.removeAttribute('id');pet.classList.add('clone');pet.setAttribute('aria-label','Subagent: '+role);const c=petColors[role]||['#6b8fb0','#daeafb'];pet.querySelector('.ghost-body').setAttribute('fill',c[0]);pet.querySelector('.ghost-spark').setAttribute('fill',c[1]);$('pet-clones').append(pet);activePets.set(role,pet)}
function mergePet(role){const pet=activePets.get(role);if(!pet)return;setTimeout(()=>{pet.classList.add('merge');setTimeout(()=>{pet.remove();activePets.delete(role)},600)},450)}
function chart(parent,title,items,metric){if(!items.length)return;const box=document.createElement('div');box.className='comparison-chart';const head=document.createElement('strong');head.textContent=title;box.append(head);const max=Math.max(...items.map(x=>Math.abs(Number(x.value))||0),.0001);items.forEach(x=>{const label=document.createElement('div');label.textContent=x.name+'：'+Number(x.value).toFixed(3)+' '+metric;const track=document.createElement('div');track.className='comparison-track';const bar=document.createElement('div');bar.className='comparison-bar';bar.style.width=Math.max(2,Math.abs(Number(x.value))/max*100)+'%';track.append(bar);box.append(label,track)});parent.append(box)}
function versionChart(parent,comparison){if(!comparison?.comparable)return;parent.dataset.comparison=JSON.stringify(comparison);const label=comparison.metric==='mae'?'MAE, lower is better':'Macro F1, higher is better';chart(parent,'Version comparison (validation set)',comparison.versions.map((v,i)=>({name:(i===0?'Previous version · ':'New version · ')+v.model,value:v.value})),label)}
function modelTable(parent,attempts){if(!attempts.length)return;const box=document.createElement('div');box.className='comparison-chart';const title=document.createElement('strong');title.textContent='Model comparison (same validation set)';box.append(title);const table=document.createElement('table');table.className='model-table';const classification=attempts[0].metric==='macro_f1';const fmt=(v,n)=>typeof v==='number'&&Number.isFinite(v)?v.toFixed(n):'—';const rows=[['Model',classification?'Macro F1 ↑':'MAE ↓',...(classification?['Accuracy ↑']:[]),'Training seconds','Inference ms / row'],...attempts.map(a=>[a.model,fmt(a.validation_score,3),...(classification?[fmt(a.validation_accuracy,3)]:[]),fmt(a.fit_seconds,2),fmt(a.predict_ms_per_row,3)])];rows.forEach((values,i)=>{const tr=document.createElement('tr');values.forEach(value=>{const cell=document.createElement(i?'td':'th');cell.textContent=value;tr.append(cell)});table.append(tr)});box.append(table);parent.append(box)}
function ablationTable(parent,a){if(!a||a.status!=='completed')return;const box=document.createElement('div');box.className='comparison-chart';const title=document.createElement('strong');title.textContent='Feature ablation: before vs after (validation set)';box.append(title);const intro=document.createElement('p');intro.textContent='Excluded: '+a.removed_features.join('、')+'; selected: '+(a.selected_profile==='reduced'?'After':'Before');box.append(intro);const table=document.createElement('table');table.className='model-table';const metric=a.metric==='mae'?'MAE ↓':'Macro F1 ↑';const rows=[['Model','Before '+metric,'After '+metric,'Before seconds','After seconds'],...a.baseline.map(before=>{const after=a.reduced.find(x=>x.model===before.model);return [before.model,Number(before.validation_score).toFixed(3),after?Number(after.validation_score).toFixed(3):'—',Number(before.fit_seconds).toFixed(2),after?Number(after.fit_seconds).toFixed(2):'—']})];rows.forEach((values,i)=>{const row=document.createElement('tr');values.forEach(v=>{const cell=document.createElement(i?'td':'th');cell.textContent=v;row.append(cell)});table.append(row)});box.append(table);const note=document.createElement('p');note.textContent=a.note;box.append(note);parent.append(box)}
function discoveryCard(parent,discovery){if(!discovery?.requested)return;const section=document.createElement('div');section.className='comparison-chart';const heading=document.createElement('strong');heading.textContent='Hugging Face model search';section.append(heading);const note=document.createElement('p');note.textContent=(discovery.status==='completed'?'Searched public models ('+(discovery.source||'source unspecified')+'）。':discovery.status==='failed'?'Model search is temporarily unavailable.':'Model search is not available for this task.')+' '+discovery.decision;section.append(note);(discovery.results||[]).slice(0,3).forEach(item=>{if(!/^https:\/\/huggingface\.co\/[A-Za-z0-9_.-]+\/[A-Za-z0-9_.-]+$/.test(item.url))return;const a=document.createElement('a');a.href=item.url;a.target='_blank';a.rel='noopener noreferrer';a.textContent=item.id+'（License: '+item.license+'）';a.style.display='block';section.append(a)});parent.append(section)}
const fileInput = $('chat-file');
if(!sessionIndex().length)createTrainingSession();else{const saved=sessionStorage.getItem('trainpilot.activeSession');const list=sessionIndex();showSession(list.some(s=>s.id===saved)?saved:list[list.length-1].id);refreshSessionSelect()}
$('new-training-session').addEventListener('click',()=>{if(sessionBusy()){dataAssistant('Finish the current task or remove the attachment before switching sessions.');return}rememberSession();createTrainingSession()});
$('training-session-select').addEventListener('change',event=>{const target=event.currentTarget.value;if(sessionBusy()){event.currentTarget.value=activeTrainingSession;dataAssistant('Finish the current task or remove the attachment before switching sessions.');return}rememberSession();showSession(target)});
let sessionSaveTimer;new MutationObserver(()=>{if(!restoringSession){clearTimeout(sessionSaveTimer);sessionSaveTimer=setTimeout(rememberSession,250)}}).observe($('chat-messages'),{childList:true,subtree:true,characterData:true});
$('request-text').addEventListener('input',()=>{clearTimeout(sessionSaveTimer);sessionSaveTimer=setTimeout(rememberSession,250)});
$('toggle-trace').addEventListener('click', () => {
  const split=document.querySelector('.console-split'); const open=split.classList.toggle('show-trace');
  $('toggle-trace').setAttribute('aria-expanded',String(open)); $('toggle-trace').textContent=open?'Hide execution log':'View execution log';
});
async function refreshGeminiBudget() {
  try {const b=await dataAPI('/api/training/agent-budget');$('gemini-budget').textContent=`${b.provider==='openai'?'OpenAI':'Gemini'} local requests available today ${b.remaining} / ${b.daily_limit} requests`;}
  catch { $('gemini-budget').textContent='Model usage is temporarily unavailable'; }
}
refreshGeminiBudget();
$('attach-file').addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => {
  $('chat-file-name').textContent = fileInput.files[0] ? 'Attached: ' + fileInput.files[0].name : 'Attach a CSV and describe your goal';
});
['preset-canonical', 'preset-followup', 'preset-pretrain'].forEach(id => $(id).addEventListener('click', () => { dataChat = null; fileInput.value = ''; $('chat-file-name').textContent = 'Attach a CSV and describe your goal'; }));

async function dataAPI(path, body) {
  activeRequests++;
  try {const headers={'X-Training-Session':activeTrainingSession};if(body!==undefined)headers['Content-Type']='application/json';
    const response = await fetch(path, body === undefined ? {headers} : {method:'POST', headers, body:JSON.stringify(body)});
    const value = await response.json();if (!response.ok) throw new Error(value.detail || 'Operation failed');return value;
  } finally {activeRequests--;}
}
function dataAssistant(text) { addWho('assistant', 'Agent'); return addMsg('assistant', d => { d.textContent = text; }); }
function followupContext(){return Array.from($('chat-messages').children).filter(e=>e.classList.contains('msg')&&(e.classList.contains('user')||e.classList.contains('assistant'))).map(e=>({role:e.classList.contains('user')?'user':'assistant',text:serializeMessage(e).text.trim().slice(0,1500)})).filter(x=>x.text).slice(-8)}
function dataButton(parent, label, handler) {
  const button = document.createElement('button');
  button.type = 'button'; button.textContent = label;
  button.style.cssText = 'margin:8px 8px 0 0;padding:7px 12px;border:1px solid #aebdca;border-radius:7px;background:#f3f8fb;color:#173b51;cursor:pointer';
  button.addEventListener('click', handler); parent.appendChild(button); return button;
}
function dataSelect(parent, title, options) {
  const label = document.createElement('label'); label.textContent = title; label.style.cssText = 'display:block;margin-top:10px;font-weight:600';
  const select = document.createElement('select'); select.style.cssText = 'display:block;max-width:100%;padding:7px;margin-top:3px';
  options.forEach(o => { const e = document.createElement('option'); e.value = o.value; e.textContent = o.label; select.appendChild(e); });
  label.appendChild(select); parent.appendChild(label); return select;
}
function showTaskChoice(info, capabilities) {
  const card = dataAssistant(`Received ${info.rows} rows. Confirm the task, target, and input features.\nColumns: ${info.columns.map(c => `${c.name} (${c.nonempty} nonempty, ${c.unique} distinct)`).join('; ')}\nRecommendations are rule-based.`);
  const task = dataSelect(card, 'Task type', [{value:'',label:'Choose a task'}, ...capabilities.tasks.map(t => ({value:t.id,label:t.label})), {value:'other',label:'Other task (describe)'}]);
  const target = dataSelect(card, 'Target column', info.columns.map(c => ({value:c.name,label:c.name})));
  if (info.columns.length > 1) target.selectedIndex = info.columns.length - 1;
  const featureTitle = document.createElement('div'); featureTitle.textContent = 'Input features (exclude identifiers, personal data, and information unavailable at prediction time)'; featureTitle.style.cssText='margin-top:10px;font-weight:600'; card.appendChild(featureTitle);
  const options = document.createElement('div'); card.appendChild(options);
  function renderFeatures() {
    options.replaceChildren();
    const capability=capabilities.tasks.find(t=>t.id===task.value);
    const needsTarget=capability ? capability.needs_target : false;
    target.parentElement.style.display = needsTarget ? '' : 'none';
    if (!capability) return;
    info.columns.forEach(c => {
      if (needsTarget && c.name === target.value) return;
      const l=document.createElement('label'); l.style.cssText='display:block;font-size:13px';
      const check=document.createElement('input');check.type='checkbox';check.value=c.name;check.checked=true;
      l.append(check,document.createTextNode(' '+c.name)); options.appendChild(l);
    });
  }
  task.addEventListener('change', renderFeatures); target.addEventListener('change', renderFeatures); renderFeatures();
  dataButton(card,'Check plan',async event => {
    const btn=event.currentTarget; btn.disabled=true;
    const capability=capabilities.tasks.find(t=>t.id===task.value);
    if (!capability) {if(task.value==='other')dataChat={id:info.id,status:'other'};dataAssistant(task.value==='other' ? 'No installed adapter supports this task. Describe the expected inputs, outputs, and evaluation.' : 'Select a task or choose Other.');btn.disabled=false;return;}
    const features=Array.from(options.querySelectorAll('input:checked'), e=>e.value);
    const spec={id:info.id, task:task.value, target:capability.needs_target?target.value:'', features};
    try {
      const plan=await dataAPI('/api/training/plan',spec); dataChat={id:info.id, spec, plan, status:'planned'};
      const advice=dataAssistant(`Suggested model: ${plan.model}\n${plan.recommendation}\nValid rows: ${plan.rows}; ${plan.target ? 'Target: '+plan.target+'\n' : ''}Input features: ${plan.features.join(', ')}\nColumn types: ${Object.entries(plan.types).map(([k,v])=>`${k}: ${v.type}, missing ${v.missing}`).join('; ')}\n${Object.keys(plan.target_summary).length ? 'Target distribution: '+JSON.stringify(plan.target_summary)+'\n' : ''}${plan.warnings.length ? 'Notes: '+plan.warnings.join('; ') : 'No blocking data issues found.'}\n${plan.task==='clustering'?'Clustering is exploratory and has no ground-truth answer.':'Results will be evaluated on a held-out test set.'}`);
      if (plan.ready) dataButton(advice,'Confirm and start training',async e=>{e.currentTarget.disabled=true;await trainDataChat(spec);});
      else dataAssistant('Adjust the data or columns as indicated and retry.');
    } catch(err){dataAssistant('Unable to create plan: '+err.message);} finally {btn.disabled=false;}
  });
}
async function trainDataChat(spec) {
  dataChat.status='training'; const progress=dataAssistant('Starting local CPU training and reload verification. Larger datasets take longer.');
  $('run-status').textContent='Training locally…'; setRunChip('running','Training locally');
  try {
    const result=await dataAPI('/api/training/train',spec); dataChat.status='completed'; dataChat.result=result;
    progress.textContent='Local training and reload verification completed.'; $('run-status').textContent='Local training completed'; setRunChip('idle','Training completed');
    const p=result.proposal, m=result.metrics;
    const metrics=Object.entries(m).map(([k,v])=>`${k}：${typeof v==='number'?v.toFixed(3):JSON.stringify(v)}`).join('\n');
    const card=dataAssistant(`${p.model}｜${p.task}\nData split: ${JSON.stringify(result.split_counts)}\n${metrics}\n\n${result.suggestions.join('\n')}\n\nAsk follow-up questions or provide a row for prediction.`);
    const link=document.createElement('a'); link.textContent='Open experiment report'; link.href=reportURL(spec.id);link.target='_blank';link.rel='noopener';link.style.cssText='display:block;margin-top:10px';card.appendChild(link);
  } catch(err) {dataChat.status='failed'; progress.textContent='Training failed: '+err.message; $('run-status').textContent='Local training failed';setRunChip('idle','Training failed');}
}
async function runDataAgent(info, clarification='') {
  const note=dataAssistant('The agent is planning and checking data and tools (up to 12 planning steps).');
  $('run-status').textContent='Agent planning…';
  const labels={model:'Analyzing request',tools:'Checking training plan',execute:'Local training',critic:'Reviewing failure',verify:'Verifying model artifact'};
  let finished=false,seen=0; $('training-popup').hidden=false; $('training-title').textContent='Preparing training';$('training-detail').textContent='Analyzing data and selecting models';$('training-progress').value=0;
  const timer=setInterval(async()=>{try{const e=await dataAPI('/api/training/'+encodeURIComponent(info.id)+'/events');const fresh=e.events.slice(seen);seen=e.events.length;fresh.forEach(ev=>{if(ev.node==='subagent'){if(ev.status==='started')splitPet(ev.role);else mergePet(ev.role)}if(ev.node==='training'){const done=ev.status==='model_complete';$('training-title').textContent='Local model comparison';$('training-detail').textContent=(ev.model||'Model')+' · '+ev.index+'/'+ev.total+(done?' completed':' training');$('training-progress').value=Math.round(((ev.index-(done?0:1))/ev.total)*100)}});const last=e.events[e.events.length-1];if(last&&!finished){note.textContent='Agent working: '+(labels[last.node]||last.node)+'（'+last.status+'）';$('run-status').textContent=note.textContent;}}catch{}},1000);
  try {
    const result=await dataAPI('/api/training/agent-run',{id:info.id,clarification});
    if (result.state !== 'completed') {
      dataChat={id:info.id,info,status:'clarify'}; note.textContent=result.message || 'Task incomplete; please provide more details.';showChoices(result.message,result.options); $('run-status').textContent=result.state === 'failed'?'Execution incomplete':'Waiting for details'; return;
    }
    if(result.model_discovery)result.result.model_discovery=result.model_discovery;
    const p=result.result.proposal; const spec={id:info.id,task:p.task,target:p.target,features:p.features};
    dataChat={id:info.id,info,plan:p,spec,status:'completed',result:result.result};
    note.textContent=`Training completed. Selected: ${p.model}。\nTask: ${p.task}；Target: ${p.target || 'none'}。`;
    if(result.specialist_findings?.length){const details=document.createElement('details');const summary=document.createElement('summary');summary.textContent='View specialist suggestions';details.appendChild(summary);result.specialist_findings.forEach(f=>{const item=document.createElement('p');item.textContent=(f.role==='data'?'Data specialist':'Model specialist')+'：'+(f.finding||'No suggestion ('+(f.error||'incomplete')+'）');details.appendChild(item)});note.appendChild(details)}
    const m=result.result.metrics; const metrics=Object.entries(m).filter(([k])=>!['per_class','labels','confusion_matrix'].includes(k)).map(([k,v])=>`${k}：${typeof v==='number'?v.toFixed(3):JSON.stringify(v)}`).join('\n');
    const cautions=result.result.suggestions||[];const hasRisk=cautions.some(x=>/識別碼|洩漏|偏樂觀|評估失真|identifier|leakage|inflate/i.test(x));
    const trained=result.result.checkpoint_reloaded?'Local models ran and the selected artifact passed reload verification. See the report for each method.':'See the report for training status.';
    const modelNote=result.result.training_type==='mixed local model comparison'?'Compared local and pretrained models; pretrained weights are not necessarily fine-tuned.':result.result.training_type==='real local scikit-learn'?'Models: local scikit-learn (no Hugging Face model loaded).':'See the report for model sources.';
    const card=dataAssistant(`${trained}\nLocal test results\n${metrics}\n${modelNote}${hasRisk?'\nNote: data or features may affect evaluation reliability; expand for details.':''}`);
    const details=document.createElement('details');const detailTitle=document.createElement('summary');detailTitle.textContent='View data split and evaluation notes';details.appendChild(detailTitle);const detailText=document.createElement('p');detailText.textContent='Data split: '+JSON.stringify(result.result.split_counts)+'\nInput features：'+p.features.join('、')+'\n'+result.result.suggestions.join('\n');details.appendChild(detailText);card.appendChild(details);
    const comparisons=(result.result.attempts||[]).filter(x=>typeof x.validation_score==='number').map(x=>({name:x.model,value:x.validation_score}));
    card.dataset.attempts=JSON.stringify((result.result.attempts||[]).filter(x=>typeof x.validation_score==='number'));
    card.dataset.ablation=JSON.stringify(result.result.ablation||null);
    card.dataset.discovery=JSON.stringify(result.model_discovery||null);
    modelTable(card,(result.result.attempts||[]).filter(x=>typeof x.validation_score==='number'));
    ablationTable(card,result.result.ablation);
    discoveryCard(card,result.model_discovery);
    if(comparisons.length)chart(card,'Model comparison (validation set)',comparisons,result.result.attempts[0].metric==='mae'?'MAE, lower is better':'Macro F1, higher is better');
    const link=document.createElement('a');link.href=reportURL(info.id);link.target='_blank';link.rel='noopener';link.textContent='Open experiment report';link.style.cssText='display:block;margin-top:10px';card.appendChild(link);
    if(info.compareWith){try{const comparison=await dataAPI('/api/training/compare?first='+encodeURIComponent(info.compareWith)+'&second='+encodeURIComponent(info.id));const extra=dataAssistant(comparison.note);if(comparison.comparable)chart(extra,'Version comparison',comparison.versions.map((v,i)=>({name:'V'+(i+1)+' · '+v.model,value:v.value})),comparison.metric);else extra.append(document.createTextNode('\n'+JSON.stringify(comparison.versions)));}catch(e){dataAssistant('Version comparison failed: '+e.message)}}
    $('run-status').textContent='Training and verification completed';$('training-title').textContent='Training and verification completed';$('training-detail').textContent='Selected model: '+p.model+'; details are in the conversation.';$('training-progress').value=100;setRunChip('idle','Training completed');
  } catch(err) {dataChat={id:info.id,info,status:'agent_error'};note.textContent='Agent execution failed: '+err.message+'. Check the execution log and retry.';$('run-status').textContent='Execution incomplete';} finally {finished=true;clearInterval(timer);refreshGeminiBudget();}
}
async function maybeHandleDataChat(text) {
  $('choice-popup').hidden=true;
  const file=fileInput.files[0];
  if (file) {
    if (file.size > 2000000) { dataAssistant('File exceeds 2 MB; use a smaller dataset.'); return true; }
    if (!text.trim()) { dataAssistant('Describe your prediction or exploration goal before uploading.'); return true; }
    $('btn-run').disabled=true; addUserBubble(text+'\nAttachment: '+file.name); $('request-text').value='';
    const note=dataAssistant('Inspecting columns, missing values, and samples…');
    try {
      const previous=dataChat&&dataChat.status==='completed'&&/比較|对比|對比|compare|v2|版本/i.test(text)?dataChat.id:null;
      const info=await dataAPI('/api/classifier/inspect',{csv:await file.text(),goal:text});if(previous)info.compareWith=previous;
      fileInput.value=''; $('chat-file-name').textContent='Attach a CSV and describe your goal';
      note.textContent=`Loaded ${info.rows} rows. Only the goal, column names, and statistics are sent to the model provider; raw rows remain local.`;
      await runDataAgent(info);
    } catch(err) {note.textContent='Unable to read data: '+err.message;} finally {$('btn-run').disabled=false;}
    return true;
  }
  const search=text.match(/^搜尋資料集[：:]\s*(.+)$/i);
  if (search) {
    addUserBubble(text); $('request-text').value=''; const note=dataAssistant('Searching public datasets using your search terms.');
    try {const res=await dataAPI('/api/datasets/search?q='+encodeURIComponent(search[1])); note.textContent=res.datasets.length?'Public datasets found. Check license, language, and suitability.':'No datasets found.';res.datasets.forEach(d=>{const a=document.createElement('a');a.href=d.url;a.target='_blank';a.rel='noopener noreferrer';a.textContent=d.id+'（License: '+d.license+'）';a.style.display='block';note.appendChild(a);});}
    catch(err){note.textContent='Search failed: '+err.message;}return true;
  }
  if (dataChat && dataChat.status === 'completed') {
    if (!text.trim()) {dataAssistant('Provide a row or upload new data with +.');return true;}
    const history=followupContext();
    addUserBubble(text); $('request-text').value='';$('btn-run').disabled=true;
    try {
      let values=null; try {values=JSON.parse(text);} catch {}
      if(values && typeof values==='object' && !Array.isArray(values) && dataChat.plan.features.every(f=>Object.prototype.hasOwnProperty.call(values,f))) {
        const result=await dataAPI('/api/training/predict',{id:dataChat.id,values});dataAssistant('Model result: '+result.value+'\n'+result.note);
      } else {
        const result=await dataAPI('/api/training/agent-chat',{id:dataChat.id,message:text,history});
        if(result.type==='variant_request'){
          const parent=dataChat;const note=dataAssistant((result.message||'Creating a new version.')+'\nComparing using the same data split. Please wait.');$('training-popup').hidden=false;$('training-title').textContent='Training new version';$('training-detail').textContent='Comparing models';$('training-progress').removeAttribute('value');
          try{const version=await dataAPI('/api/training/variant',{parent_id:parent.id,features:result.features,options:result.options});dataChat={id:version.id,info:parent.info,plan:version.result.proposal,spec:parent.spec,status:'completed',result:version.result};note.textContent=(result.message||'')+'\nNew version completed and passed reload verification.\n'+version.comparison.note;note.dataset.attempts=JSON.stringify((version.result.attempts||[]).filter(x=>typeof x.validation_score==='number'));modelTable(note,(version.result.attempts||[]).filter(x=>typeof x.validation_score==='number'));versionChart(note,version.comparison);const a=document.createElement('a');a.href=reportURL(version.id);a.target='_blank';a.rel='noopener';a.textContent='Open new version report';a.style.display='block';note.append(a);$('training-title').textContent='New version completed';$('training-detail').textContent='Model comparison and verification completed';$('training-progress').value=100;$('run-status').textContent='New version training completed';}
          catch(err){note.textContent='New version did not complete：'+err.message;$('training-title').textContent='New version did not complete';$('training-detail').textContent='Training stopped. See the error in the conversation.';$('training-progress').value=0;$('run-status').textContent='New version training failed';}
        }else {dataAssistant(result.message);if(result.type==='clarify')showChoices(result.message,result.options);}
      }
    } catch(err){dataAssistant('Unable to complete: '+err.message);}finally{$('btn-run').disabled=false;refreshGeminiBudget();}return true;
  }
  if (dataChat && (dataChat.status === 'clarify' || dataChat.status === 'agent_error')) {
    if (!text.trim()) {dataAssistant('Add details or upload different data.');return true;}
    addUserBubble(text);$('request-text').value='';$('btn-run').disabled=true;try{await runDataAgent(dataChat.info,text)}finally{$('btn-run').disabled=false}return true;
  }
  if (dataChat && dataChat.status === 'other') {addUserBubble(text);$('request-text').value='';dataAssistant('No installed adapter supports this request yet. Add an adapter and validate the data and evaluation method first.');return true;}
  if (dataChat && dataChat.status !== 'completed') {dataAssistant('Complete the task and feature selection above, or upload new data with +.');return true;}
  if (!text.trim()) {dataAssistant('Describe your training goal or upload a CSV with +.');return true;}
  addUserBubble(text);$('request-text').value='';intakeHistory.push({role:'user',text});intakeHistory=intakeHistory.slice(-7);
  const note=dataAssistant('Thinking…');$('btn-run').disabled=true;
  try {const answer=await dataAPI('/api/training/agent-intake',{messages:intakeHistory});note.textContent=answer.message;showChoices(answer.message,answer.options);(answer.datasets||[]).slice(0,4).forEach(d=>{const a=document.createElement('a');a.href=d.url;a.target='_blank';a.rel='noopener noreferrer';a.textContent=d.id||d.ref;a.style.display='block';note.appendChild(a)});intakeHistory.push({role:'assistant',text:answer.message.slice(0,1000)});if(answer.prepared?.state==='prepared'){const prepared=answer.prepared;previousChoices=null;dataAssistant('Source: '+prepared.source.url+'\nData: '+prepared.source.filename+'，'+prepared.dataset.rows+' rows. Starting local analysis and training.');await runDataAgent(prepared.dataset);}}
  catch(err){note.textContent='Unable to complete: '+err.message;if(previousChoices&&/^使用 Kaggle (資料集|競賽) /.test(text))showChoices('Unable to read the selected source; choose another.',previousChoices.options);}finally{$('btn-run').disabled=false;refreshGeminiBudget();}
  return true;
}

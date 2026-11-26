const $=id=>document.getElementById(id);let chosen=null,lastId=null,chosenSecondary=null,primaryImageUrl=null,secondaryImageUrl=null;
const setBadge=(id,txt)=>{const el=$(id);el.textContent=txt;el.className=txt==='PROVISIONAL_PASS'?'good':txt==='FAIL'?'fail':'unknown'};
function setImage(f){if(!f)return;$('formal-report').hidden=true;$('print-report').disabled=true;lastId=null;chosen=f;$('filename').textContent=`${f.name} · ${(f.size/1024/1024).toFixed(2)} MB`;$('analyze').disabled=false;if(primaryImageUrl)URL.revokeObjectURL(primaryImageUrl);primaryImageUrl=URL.createObjectURL(f);$('preview').src=primaryImageUrl;$('preview').hidden=false;$('placeholder').hidden=true}
$('file').addEventListener('change',e=>setImage(e.target.files[0]));
$('secondary-file').addEventListener('change',e=>{chosenSecondary=e.target.files[0]||null;if(secondaryImageUrl)URL.revokeObjectURL(secondaryImageUrl);secondaryImageUrl=chosenSecondary?URL.createObjectURL(chosenSecondary):null;$('secondary-filename').textContent=chosenSecondary?chosenSecondary.name+' • Display-only, not model input':'No extra view selected. This image is NOT analyzed by the AI.';});
const drop=$('drop-zone');drop.addEventListener('dragover',e=>{e.preventDefault();drop.classList.add('drag')});drop.addEventListener('dragleave',()=>drop.classList.remove('drag'));drop.addEventListener('drop',e=>{e.preventDefault();drop.classList.remove('drag');setImage(e.dataTransfer.files[0])});
$('analyze').addEventListener('click',async()=>{if(!chosen)return;const btn=$('analyze');btn.disabled=true;btn.textContent='Analyzing research image…';
try{const data=new FormData();data.append('file',chosen);data.append('projection',$('projection').value);const res=await fetch('/api/analyze',{method:'POST',body:data});const out=await res.json();if(!res.ok)throw Error(out.error||'Analysis failed');lastId=out.id;
$('report').textContent=out.draft_impression||'NO INFERENCE: a trained report-generator checkpoint is not installed. The system has NOT interpreted this X-ray.';
$('report-status').textContent=out.model_loaded?'RESEARCH DRAFT':'UNTRAINED';
renderFormal(out.formal_report);
setBadge('entity',out.safety.entity_check.status);setBadge('consistency',out.safety.self_consistency===null?'UNKNOWN':Math.round(out.safety.self_consistency*100)+'%');setBadge('gradcam',out.safety.grounding.status);
$('safety-details').textContent='Entity: '+(out.safety.entity_check.reason||'')+'\nContradictions: '+JSON.stringify(out.safety.entity_check.contradictions||out.safety.entity_check.claims?.contradictions||[])+'\nConsistency: '+(out.safety.self_consistency_details?.warning||'')+'\nGrad-CAM: '+(out.safety.grounding.reason||'')+'\nDraft choice: '+(out.draft_selection?.reason||'Not available');
$('review-flag').textContent='MANDATORY HUMAN REVIEW • '+out.safety.review_reasons.join(' • ');
$('heatmap-container').hidden=!out.gradcam_overlay;if(out.gradcam_overlay){
    $('heatmap').src=out.gradcam_overlay;
    const label=out.safety.grounding.preview_label || 'Classifier Grad-CAM';
    $('heatmap').alt=label;
    $('heatmap-container').querySelector('h3').textContent=label;
}else{$('heatmap').removeAttribute('src');}
$('review-form').hidden=!out.model_loaded;await refreshQueue();
}catch(e){$('report').textContent='Error: '+e.message;renderFormal(null);}finally{btn.disabled=false;btn.innerHTML='Analyze X-ray <span>→</span>'}});
async function refreshQueue(){try{const res=await fetch('/api/reviews');const d=await res.json();$('queue').innerHTML=d.reviews.length?d.reviews.map(v=>`<div class="queue-row"><div><strong>${v.id.slice(0,12)}…</strong><small>${new Date(v.timestamp).toLocaleString()} · ${v.model_version.replace(/[<>&"']/g,'')}</small></div><span class="tag">${v.state}</span></div>`).join(''):'No review records yet.';}catch(e){$('queue').textContent='Queue unavailable: '+e.message}}
$('refresh').addEventListener('click',refreshQueue);
$('signoff').addEventListener('click',async()=>{if(!lastId)return;try{const r=await fetch(`/api/reviews/${lastId}/decision`,{method:'POST',headers:{'Content-Type':'application/json','X-Review-Token':$('token').value},body:JSON.stringify({reviewer:$('reviewer').value,decision:$('decision').value})});const d=await r.json();if(!r.ok)throw Error(d.error||'Review failed');$('review-msg').textContent='Simulated review decision recorded; not clinical approval.';await refreshQueue();}catch(e){$('review-msg').textContent=e.message}});
async function status(){try{const r=await fetch('/model-info');const d=await r.json();$('mode-status').textContent=d.model_loaded?'MODEL CHECKPOINT LOADED':'DEMO / NO MODEL';$('model-hint').textContent=d.model_loaded?'Research model loaded. Findings and labels remain unverified until qualified review.':'No trained checkpoint? This interface never invents a radiology impression.';}catch(e){$('mode-status').textContent='UNAVAILABLE'}}status();refreshQueue();

function fillReportLines(id, items, empty){
  const host=$(id);host.replaceChildren();
  if(!Array.isArray(items)||items.length===0){const label=document.createElement('span');label.className='annotation-empty';label.textContent=empty;host.append(label);return;}
  for(const item of items){const tag=document.createElement('span');tag.className='report-chip';
    tag.textContent=typeof item==='string'?item:(item.name+' · '+item.polarity.toLowerCase().replaceAll('_',' '));
    host.append(tag);
  }
}
function renderFormal(r){
  if(!r){$('formal-report').hidden=true;$('print-report').disabled=true;return;}
  const fields={'r-id':'report_id','r-date':'created_utc','r-patient':'patient_details',
    'r-version':'model_version','r-examination':'examination','r-history':'clinical_history',
    'r-technique':'technique','r-comparison':'comparison','r-findings':'findings',
    'r-impression':'impression','r-radiologist':'radiologist','r-disclaimer':'disclaimer',
    'formal-title':'report_title'};
  for(const [id,field] of Object.entries(fields))$(id).textContent=r[field]||'Not available';
  $('r-study').textContent=chosen?.name||'No image selected';
  $('r-primary-image').src=primaryImageUrl||$('preview').src;
  $('r-primary-caption').textContent='Primary X-ray • '+($('projection').selectedOptions[0]?.textContent||'projection unknown')+' • analyzed by model';
  $('r-secondary-figure').hidden=!chosenSecondary;
  if(chosenSecondary){$('r-secondary-image').src=secondaryImageUrl;$('r-secondary-caption').textContent=chosenSecondary.name+' • display only (NOT analyzed)';}
  fillReportLines('r-observations',r.observations,'No positive observations extracted from this AI draft; this does not establish a normal X-ray.');
  fillReportLines('r-labels',r.labels,'No finding labels extracted from generated text.');
  fillReportLines('r-mesh',r.mesh_terms,'No suggested MeSH terms from positive draft findings.');
  fillReportLines('r-image-signals',(r.image_signals||[]).map(x=>x.name+' · research score '+Math.round(x.score*100)+'%'),
      'No above-threshold image classifier signals available; not a normal-image verdict.');
  $('r-normal').textContent=r.normal_statement||r.annotations_note||'';
  $('r-image-note').textContent=r.image_signals_note||'Not a medically validated image classifier.';
  $('r-qc').textContent=(r.review_status||'MANDATORY_REVIEW')+' — '+((r.quality_flags||[]).join(' • ')||'All research drafts require qualified review.');
  $('r-source').textContent='Text source: '+(r.report_source||'UNKNOWN')+' • Findings and Impression sections complete: '+(r.both_sections_generated?'yes':'no');
  $('formal-report').hidden=false;
  $('print-report').disabled=!r.impression || r.report_source==='NO_INFERENCE';
}
$('print-report').addEventListener('click',()=>window.print());

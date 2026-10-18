"""Constructor-level metadata mutations; copies only five sidecars, never weights."""
import json,pathlib,shutil,subprocess,tempfile

def run(binary,source):
    cases=[
        ('config.json','mlp_bias true',lambda x:x.update(mlp_bias=True)),
        ('config.json','mlp_bias wrong type',lambda x:x.update(mlp_bias=0)),
        ('config.json','head_dim128',lambda x:x.update(head_dim=128)),
        ('config.json','head_dim float',lambda x:x.update(head_dim=64.0)),
        ('config.json','partial_rotary override',lambda x:x.update(partial_rotary_factor=.5)),
        ('config.json','rope override',lambda x:x.update(rope_scaling={'type':'linear','factor':2})),
        ('config.json','architecture NUL',lambda x:x.update(model_type='llama\0other')),
        ('special_tokens_map.json','bos lstrip',lambda x:x['bos_token'].update(lstrip=True)),
        ('special_tokens_map.json','eos normalized',lambda x:x['eos_token'].update(normalized=True)),
        ('special_tokens_map.json','unk rstrip',lambda x:x['unk_token'].update(rstrip=True)),
        ('special_tokens_map.json','pad single_word',lambda x:x['pad_token'].update(single_word=True)),
        ('special_tokens_map.json','additional tokens',lambda x:x.update(additional_special_tokens=['<new>'])),
        ('special_tokens_map.json','unknown token arrangement',lambda x:x.update(mask_token={'content':'<mask>'})),
        ('tokenizer_config.json','additional tokens',lambda x:x.update(additional_special_tokens=['<new>'])),
        ('tokenizer_config.json','additional tokens wrong type',lambda x:x.update(additional_special_tokens='')),
        ('tokenizer_config.json','unknown normalization',lambda x:x.update(do_lower_case=True)),
        ('tokenizer_config.json','added token normalized',lambda x:x['added_tokens_decoder']['0'].update(normalized=True)),
        ('tokenizer_config.json','template NUL',lambda x:x.update(chat_template='a\0b')),
        ('tokenizer_config.json','special content NUL',lambda x:x.update(bos_token='<s>\0')),
    ]
    with tempfile.TemporaryDirectory(prefix='original-sidecars-') as folder:
        folder=pathlib.Path(folder)
        for name in ('config.json','tokenizer.model','tokenizer_config.json','tokenizer.json','special_tokens_map.json'):
            shutil.copyfile(source/name,folder/name)
        def invoke(expected):
            result=subprocess.run([str(binary),'--construct',str(folder)],capture_output=True,text=True,timeout=15)
            assert result.returncode==expected,(result.returncode,result.stderr)
            assert not any(marker in result.stderr for marker in ('AddressSanitizer','UndefinedBehaviorSanitizer','LeakSanitizer','runtime error:')),(result.returncode,result.stderr)
            if expected:assert result.stderr.startswith('REJECT ') and not result.stdout,(result.stdout,result.stderr)
            else:assert result.stdout=='CONSTRUCTOR_PASS\n' and not result.stderr,(result.stdout,result.stderr)
        invoke(0)
        for name,label,mutate in cases:
            path=folder/name;original=path.read_bytes();value=json.loads(original);mutate(value)
            try:
                path.write_text(json.dumps(value));invoke(1)
            except Exception as e:raise AssertionError(label) from e
            finally:path.write_bytes(original)
        for name,update in [('config.json',{'head_dim':64,'mlp_bias':False}),('tokenizer_config.json',{'additional_special_tokens':[]}),('special_tokens_map.json',{'additional_special_tokens':[]})]:
            path=folder/name;value=json.loads(path.read_bytes());value.update(update);path.write_text(json.dumps(value))
        invoke(0)
    print(f'PASS constructor mutations: {len(cases)} rejected, original and supported optional forms accepted')

def generate(binary,folder):
    """Produce small valid fixed-profile sidecars; no downloaded model inputs."""
    folder.mkdir()
    subprocess.run([str(binary),'--write-synthetic',str(folder/'tokenizer.model')],check=True)
    config=dict(architectures=['LlamaForCausalLM'],model_type='llama',hidden_act='silu',attention_bias=False,tie_word_embeddings=False,rope_scaling=None,torch_dtype='bfloat16',bos_token_id=1,eos_token_id=2,num_hidden_layers=22,max_position_embeddings=2048,hidden_size=2048,intermediate_size=5632,num_attention_heads=32,num_key_value_heads=4,vocab_size=32000,pretraining_tp=1,rms_norm_eps=1e-5,rope_theta=10000.0)
    pieces=['<unk>','<s>','</s>']+[f'<0x{i:02X}>' for i in range(256)]+[f'word{i}' for i in range(259,32000)]
    flags=dict(lstrip=False,rstrip=False,normalized=False,single_word=False)
    added={str(i):dict(content=pieces[i],special=True,**flags) for i in range(3)}
    tc=dict(tokenizer_class='LlamaTokenizer',clean_up_tokenization_spaces=False,legacy=False,sp_model_kwargs={},use_default_system_prompt=False,padding_side='right',model_max_length=2048,added_tokens_decoder=added,chat_template='{{ messages }}',bos_token='<s>',eos_token='</s>',unk_token='<unk>',pad_token='</s>')
    special={name:dict(content=tc[name],**flags) for name in ('bos_token','eos_token','unk_token','pad_token')}
    tj=dict(model=dict(type='BPE',byte_fallback=True,dropout=None,unk_token='<unk>',fuse_unk=True,continuing_subword_prefix=None,end_of_word_suffix=None,vocab={p:i for i,p in enumerate(pieces)}),added_tokens=[dict(id=i,**added[str(i)]) for i in range(3)],pre_tokenizer=None,
        normalizer=dict(type='Sequence',normalizers=[dict(type='Prepend',prepend='\u2581'),dict(type='Replace',pattern=dict(String=' '),content='\u2581')]),
        decoder=dict(type='Sequence',decoders=[dict(type='Replace',pattern=dict(String='\u2581'),content=' '),dict(type='ByteFallback'),dict(type='Fuse'),dict(type='Strip',content=' ',start=1,stop=0)]),
        post_processor=dict(type='TemplateProcessing',special_tokens={'<s>':dict(id='<s>',ids=[1],tokens=['<s>'])},single=[dict(SpecialToken=dict(id='<s>',type_id=0)),dict(Sequence=dict(id='A',type_id=0))]))
    for name,value in [('config.json',config),('tokenizer_config.json',tc),('special_tokens_map.json',special),('tokenizer.json',tj)]:
        (folder/name).write_text(json.dumps(value),encoding='utf-8')

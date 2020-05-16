Search.setIndex({docnames:["README","api","cmd","index","schema"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":2,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":2,"sphinx.domains.rst":2,"sphinx.domains.std":1,"sphinx.ext.intersphinx":1,"sphinx.ext.todo":2,"sphinx.ext.viewcode":1,sphinx:56},filenames:["README.rst","api.rst","cmd.rst","index.rst","schema.rst"],objects:{"narchi.blocks":{ConcatBlocksEnum:[1,1,1,""],ConvBlocksEnum:[1,1,1,""],FixedOutputBlocksEnum:[1,1,1,""],GroupPropagatorsEnum:[1,1,1,""],ReshapeBlocksEnum:[1,1,1,""],RnnBlocksEnum:[1,1,1,""],SameShapeBlocksEnum:[1,1,1,""],register_known_propagators:[1,3,1,""],register_propagator:[1,3,1,""]},"narchi.blocks.ConcatBlocksEnum":{Concatenate:[1,2,1,""]},"narchi.blocks.ConvBlocksEnum":{AvgPool1d:[1,2,1,""],AvgPool2d:[1,2,1,""],AvgPool3d:[1,2,1,""],Conv1d:[1,2,1,""],Conv2d:[1,2,1,""],Conv3d:[1,2,1,""],MaxPool1d:[1,2,1,""],MaxPool2d:[1,2,1,""],MaxPool3d:[1,2,1,""]},"narchi.blocks.FixedOutputBlocksEnum":{AdaptiveAvgPool2d:[1,2,1,""],Linear:[1,2,1,""]},"narchi.blocks.GroupPropagatorsEnum":{Group:[1,2,1,""],Module:[1,2,1,""],Sequential:[1,2,1,""]},"narchi.blocks.ReshapeBlocksEnum":{Reshape:[1,2,1,""]},"narchi.blocks.RnnBlocksEnum":{GRU:[1,2,1,""],LSTM:[1,2,1,""],RNN:[1,2,1,""]},"narchi.blocks.SameShapeBlocksEnum":{Add:[1,2,1,""],BatchNorm2d:[1,2,1,""],Dropout:[1,2,1,""],Identity:[1,2,1,""],LeakyReLU:[1,2,1,""],LogSigmoid:[1,2,1,""],LogSoftmax:[1,2,1,""],ReLU:[1,2,1,""],Sigmoid:[1,2,1,""],Softmax:[1,2,1,""],Tanh:[1,2,1,""]},"narchi.graph":{parse_graph:[1,3,1,""]},"narchi.instantiators":{common:[1,0,0,"-"],pytorch:[1,0,0,"-"]},"narchi.instantiators.common":{id_strip_parent_prefix:[1,3,1,""],import_object:[1,3,1,""],instantiate_block:[1,3,1,""]},"narchi.instantiators.pytorch":{Add:[1,1,1,""],BaseModule:[1,1,1,""],Group:[1,1,1,""],Reshape:[1,1,1,""],Sequential:[1,1,1,""],StandardModule:[1,1,1,""],graph_forward:[1,3,1,""]},"narchi.instantiators.pytorch.Add":{forward:[1,4,1,""]},"narchi.instantiators.pytorch.BaseModule":{__init__:[1,4,1,""],forward:[1,4,1,""],state_dict_prop:[1,4,1,""]},"narchi.instantiators.pytorch.Group":{__init__:[1,4,1,""],forward:[1,4,1,""]},"narchi.instantiators.pytorch.Reshape":{__init__:[1,4,1,""],forward:[1,4,1,""]},"narchi.instantiators.pytorch.Sequential":{__init__:[1,4,1,""]},"narchi.instantiators.pytorch.StandardModule":{blocks_mappings:[1,2,1,""]},"narchi.module":{ModuleArchitecture:[1,1,1,""],ModulePropagator:[1,1,1,""]},"narchi.module.ModuleArchitecture":{__init__:[1,4,1,""],apply_config:[1,4,1,""],architecture:[1,2,1,""],blocks:[1,2,1,""],get_config_parser:[1,4,1,""],jsonnet:[1,2,1,""],load_architecture:[1,4,1,""],path:[1,2,1,""],propagate:[1,4,1,""],propagators:[1,2,1,""],topological_predecessors:[1,2,1,""],validate:[1,4,1,""],write_json:[1,4,1,""],write_json_outdir:[1,4,1,""]},"narchi.module.ModulePropagator":{num_input_blocks:[1,2,1,""],propagate:[1,4,1,""]},"narchi.propagators":{base:[1,0,0,"-"],conv:[1,0,0,"-"],fixed:[1,0,0,"-"],group:[1,0,0,"-"],reshape:[1,0,0,"-"],rnn:[1,0,0,"-"],same:[1,0,0,"-"]},"narchi.propagators.base":{BasePropagator:[1,1,1,""],check_output_feats_dims:[1,3,1,""],create_shape:[1,3,1,""],get_shape:[1,3,1,""],set_shape_dim:[1,3,1,""],shape_has_auto:[1,3,1,""],shapes_agree:[1,3,1,""]},"narchi.propagators.base.BasePropagator":{__call__:[1,4,1,""],__init__:[1,4,1,""],block_class:[1,2,1,""],final_checks:[1,4,1,""],initial_checks:[1,4,1,""],num_input_blocks:[1,2,1,""],output_feats_dims:[1,2,1,""],propagate:[1,4,1,""]},"narchi.propagators.conv":{ConvPropagator:[1,1,1,""],PoolPropagator:[1,1,1,""]},"narchi.propagators.conv.ConvPropagator":{__init__:[1,4,1,""],conv_dims:[1,2,1,""],initial_checks:[1,4,1,""],num_features_source:[1,2,1,""],num_input_blocks:[1,2,1,""],propagate:[1,4,1,""]},"narchi.propagators.conv.PoolPropagator":{num_features_source:[1,2,1,""]},"narchi.propagators.fixed":{FixedOutputPropagator:[1,1,1,""]},"narchi.propagators.fixed.FixedOutputPropagator":{__init__:[1,4,1,""],initial_checks:[1,4,1,""],num_input_blocks:[1,2,1,""],output_feats_dims:[1,2,1,""],propagate:[1,4,1,""],unfixed_dims:[1,2,1,""]},"narchi.propagators.group":{GroupPropagator:[1,1,1,""],SequentialPropagator:[1,1,1,""],add_ids_prefix:[1,3,1,""],get_blocks_dict:[1,3,1,""],propagate_shapes:[1,3,1,""]},"narchi.propagators.group.GroupPropagator":{propagate:[1,4,1,""]},"narchi.propagators.group.SequentialPropagator":{num_input_blocks:[1,2,1,""],propagate:[1,4,1,""]},"narchi.propagators.reshape":{ReshapePropagator:[1,1,1,""],check_reshape_spec:[1,3,1,""],norm_reshape_spec:[1,3,1,""]},"narchi.propagators.reshape.ReshapePropagator":{initial_checks:[1,4,1,""],num_input_blocks:[1,2,1,""],propagate:[1,4,1,""]},"narchi.propagators.rnn":{RnnPropagator:[1,1,1,""]},"narchi.propagators.rnn.RnnPropagator":{initial_checks:[1,4,1,""],num_input_blocks:[1,2,1,""],output_feats_dims:[1,2,1,""],propagate:[1,4,1,""]},"narchi.propagators.same":{SameShapePropagator:[1,1,1,""],SameShapesPropagator:[1,1,1,""]},"narchi.propagators.same.SameShapePropagator":{initial_checks:[1,4,1,""],propagate:[1,4,1,""]},"narchi.propagators.same.SameShapesPropagator":{num_input_blocks:[1,2,1,""]},"narchi.render":{ModuleArchitectureRenderer:[1,1,1,""]},"narchi.render.ModuleArchitectureRenderer":{apply_config:[1,4,1,""],create_graph:[1,4,1,""],get_config_parser:[1,4,1,""],render:[1,4,1,""]},"narchi.schemas":{schema_as_str:[1,3,1,""]},"narchi.sympy":{divide:[1,3,1,""],get_nonrational_variable:[1,3,1,""],is_valid_dim:[1,3,1,""],prod:[1,3,1,""],sum:[1,3,1,""],sympify_variable:[1,3,1,""],variable_operate:[1,3,1,""],variables_aggregate:[1,3,1,""]},narchi:{blocks:[1,0,0,"-"],graph:[1,0,0,"-"],module:[1,0,0,"-"],render:[1,0,0,"-"],schemas:[1,0,0,"-"],sympy:[1,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","attribute","Python attribute"],"3":["py","function","Python function"],"4":["py","method","Python method"]},objtypes:{"0":"py:module","1":"py:class","2":"py:attribute","3":"py:function","4":"py:method"},terms:{"5c106cde":[0,3],"case":1,"class":[0,1,3],"const":[1,4],"default":2,"enum":[1,4],"function":[0,1,3],"import":[0,1,3],"int":1,"new":[0,3],"return":1,"static":1,"true":[1,2],"while":[0,1,3],IDs:[1,2],One:1,That:1,The:[0,1,2,3],__call__:1,__init__:1,__input__:1,_class:[0,1,3,4],_descript:4,_ext_var:4,_id:[0,1,3,4],_id_shar:4,_name:[0,3,4],_path:4,_shape:[1,4],accord:1,acycl:1,adapt:1,adaptiveavgpool2d:1,add:[0,1,2,3],add_ids_prefix:1,additionalproperti:4,after:1,afterward:1,against:[1,2],aggreg:1,agraph:1,agre:1,all:1,allof:4,allow:[0,3],along:1,alreadi:[1,2],also:[0,3],although:1,alwai:1,among:1,ani:1,anoth:1,api:3,appli:1,apply_config:1,approach:[0,3],architectur:[1,2,4],arg:1,argumentpars:1,arithmet:[0,3],arrai:[1,4],attribut:[1,2],auto:[1,4],automat:[0,3],avail:2,averag:1,avgpool1d:1,avgpool2d:1,avgpool3d:1,bare:[0,3],base:1,basemodul:1,basepropag:1,basic:[0,3],batch:1,batch_first:1,batchnorm2d:[0,1,3],been:[1,2],befor:1,being:[0,1,3],below:[0,3],between:[0,3],bia:[0,3],bidirect:1,block:[0,2,3,4],block_attr:2,block_cfg:1,block_class:1,block_id:1,block_label:2,blocks_dict:1,blocks_map:1,bn1:[0,3],bn2:[0,3],bool:1,both:1,box:2,call:1,can:[0,3],care:1,cfg:[0,1,2,3],check:[1,2],check_output_feats_dim:1,check_reshape_spec:1,choic:2,circl:2,circo:2,code:1,com:4,command:[0,3],common:1,complet:1,comput:1,concat:1,concatblocksenum:1,concaten:[1,4],concatenatepropag:1,config:1,configur:[0,1,2,3],connect:1,consist:1,contain:1,conv1:[0,3],conv1d:1,conv2:[0,3],conv2d:[0,1,3],conv3d:1,conv:1,conv_dim:1,convblocksenum:1,convert:1,convolut:1,convpropag:1,copi:1,could:1,creat:[0,1,3],create_graph:1,create_shap:1,crop:[0,3],current:[1,2],cwd:[1,2],dash:2,deduc:[0,3],defin:[0,1,3],definit:[1,4],denomin:1,depth:2,deriv:[0,1,3],descript:4,detail:[0,3],determin:1,diagram:[0,1,2,3],dict:1,dictionari:1,dilat:[0,3],dim:[1,4],dimens:1,direct:1,directori:[1,2],divid:1,divis:1,doe:1,don:1,done:[0,3],dot:[1,2],downsampl:[0,3],draft:4,draw:2,dropout:1,each:[0,1,3],eas:[0,3],easier:[0,3],easili:[0,3],edg:2,edge_attr:2,effort:[0,3],either:1,element:1,els:4,error:[0,3],even:1,everi:1,exactli:1,exist:[0,2,3],exit:2,expect:1,express:1,ext_var:[0,1,2,3],extens:[1,2],extern:[1,2],fact:1,fail:1,fals:[0,1,2,3,4],fdp:2,file:[0,1,2,3],fill:2,final_check:1,first:[0,3],fix:[1,2],fixed_dim:1,fixedoutputblocksenum:1,fixedoutputpropag:1,flatten:1,follow:[0,1,3],fontsiz:2,format:[0,1,2,3],former:1,forward:[0,1,3],found:1,free:1,from:[0,1,3],from_block:1,from_shap:1,full_id:2,gener:[0,1,3],get:1,get_blocks_dict:1,get_config_pars:1,get_nonrational_vari:1,get_shap:1,given:1,glock:1,graph:[0,3,4],graph_forward:1,graphviz:2,group:[0,1,3,4],grouppropag:1,grouppropagatorsenum:1,gru:1,has:[1,2],have:[0,1,3],here:[0,3],hexagon:2,highli:[0,3],hook:1,http:4,hyperbol:1,id_strip_parent_prefix:1,ident:[0,1,2,3],identifi:2,ignor:1,illustr:[0,3],implement:[0,1,3],import_object:1,in_channel:1,in_featur:1,includ:[0,1,2,3],independ:[0,3],index:[1,3],inform:[0,3],initi:1,initial_check:1,inner:2,input:[0,1,2,3,4],input_s:1,inputs_output:4,instanc:1,instanti:[0,3],instantiate_block:1,instead:1,integ:4,intend:[0,3],intern:[0,1,3],invalid:1,involv:1,io_block:1,is_valid_dim:1,item:4,its:1,json:[0,1,2,3],json_path:1,jsonargpars:1,jsonnet:[0,1,2,3,4],jsonnet_path:2,kei:1,kernel_s:[0,3],kwarg:1,label:2,last:[1,2],latter:1,layer2:[0,3],layer:[0,3],layout:2,layout_prog:2,leaki:1,leakyrelu:1,least:1,len:1,less:[0,3],like:[0,3],line:[0,3],linear:1,list:1,load:1,load_architectur:1,log:1,logsigmoid:1,logsoftmax:1,lstm:1,main:2,make:[0,1,3],map:[1,2],margin:2,maximum:[1,2],maxpool1d:1,maxpool2d:1,maxpool3d:1,maxproperti:4,method:[1,2],mind:[0,3],minimum:4,minitem:4,minlength:4,minproperti:4,modul:[0,2,3,4],modular:[0,3],module_cfg:1,modulearchitectur:1,modulearchitecturerender:1,modulepropag:1,most:1,multi:1,multi_input:1,multipl:1,must:1,name:1,namespac:1,narchi:[2,4],narchi_cli:[0,3],neato:2,need:1,nest:2,nested_depth:[0,2,3],network:[1,2,4],neural:[1,2,4],node:[1,2],none:[1,2],nor:1,norm_reshape_spec:1,normal:1,note:[0,3],noth:1,notimplementederror:1,num_block:[0,3],num_featur:1,num_features_sourc:1,num_input_block:1,number:[1,2],numer:1,object:[1,4],omniu:4,one:1,oneof:4,onli:1,oper:1,option:1,order:1,ordereddict:1,org:4,other:[0,1,3],out:[1,4],out_channel:1,out_featur:1,out_fil:2,out_gv:[],out_id:1,out_index:1,out_json:[],out_rend:1,outdir:2,output:[0,1,3,4],output_feat:1,output_feats_dim:1,output_s:[0,1,3],over:1,overridden:1,overwrit:2,packag:1,pad:[0,3],page:3,paramet:[0,1,3],parent:[1,2],parent_id:2,pars:[1,2],parse_graph:1,parser:1,part:[0,3],pass:1,path:[1,2,4],pattern:[1,4],patternproperti:4,pdf:[0,2,3],penwidth:2,perform:1,peripheri:2,permut:1,pool:1,poolpropag:1,possibl:2,potenti:[0,3],prefix:[1,2],present:1,preserv:1,pretti:[1,2],previou:[0,3],print:[1,2],problem:1,prod:1,product:1,program:2,prone:[0,3],propag:[0,2,3],propagate_shap:1,properti:[1,4],provid:[0,3],pth:[0,3],pygraphviz:[1,2],python:[0,3],pytorch:[0,1,3],rais:1,randomli:1,rang:1,receiv:1,recip:1,rectifi:1,recurr:1,ref:4,refer:[0,3],referenc:2,regist:1,register_known_propag:1,register_propag:1,reimplement:[0,3],rel:1,relat:1,relationship:[0,3],relu1:[0,3],relu2:[0,3],relu:1,remov:1,render:[0,3],repeat:[0,3],replac:1,requir:[0,1,2,3,4],resblock:[0,3],reshap:[1,2,4],reshape_dim:4,reshape_flatten:4,reshape_index:4,reshape_spec:[1,4],reshape_unflatten:4,reshapeblocksenum:1,reshapepropag:1,resnet18:[0,3],resnet:[0,3],resolv:1,respect:[0,1,3],result:1,rnn:1,rnnblocksenum:1,rnnpropag:1,round:2,run:1,same:1,sameshapeblocksenum:1,sameshapepropag:1,sameshapespropag:1,save:2,save_gv:2,save_json:2,save_pdf:2,schema:3,schema_as_str:1,scriptmodul:1,search:3,see:[0,3],seen:[0,3],separ:1,sequenc:1,sequenti:[0,1,3,4],sequentialpropag:1,set:[1,2],set_shape_dim:1,shape:[0,1,2,3,4],shape_from:1,shape_has_auto:1,shape_in:1,shape_out:1,shape_to:1,shapes_agre:1,share:[1,2],should:[1,2],show:2,sigmoid:1,silent:1,simpl:[0,1,3],simplenamespac:1,sinc:[0,1,3],singl:[0,1,3],size:1,skip:1,skip_id:1,skip_io:1,small:[0,3],softmax:1,some:1,sort:1,sourc:1,specif:1,standard:1,standardmodul:[0,1,3],start:[0,3],state:1,state_dict:[0,1,3],state_dict_prop:1,statement:1,step:2,str:1,stride:[0,3],string:[1,4],style:[1,2],subblock:2,subclass:1,success:2,sum:1,sure:1,symbol:[0,1,3],sympi:3,sympify_vari:1,sympyfi:1,take:1,tangent:1,tanh:1,task:[0,3],tensor:[0,1,3],them:1,thi:[0,1,3],though:[0,3],through:1,thu:[0,1,3],titl:4,tool:[0,3],topolog:1,topological_predecessor:1,torch:1,torchvis:[0,3],transform:1,two:1,twopi:2,type:[1,4],undefin:1,understand:[0,3],unfix:1,unfixed_dim:1,unflatten:1,unit:1,unlimit:2,unset:2,usag:2,use:2,useful:[0,1,3],uses:2,using:[0,1,3],val:1,valid:[0,1,3],valu:1,valueerror:1,variabl:[1,2,4],variable_oper:1,variable_regex:1,variables_aggreg:1,version:2,wai:[0,3],what:[0,3],when:1,where:[1,2],whether:[1,2],which:[0,1,2,3],width:2,wise:1,within:1,without:[0,3],work:[1,2],write:[0,1,2,3],write_json:1,write_json_outdir:1,written:[0,3],yes:2,you:[0,3],zero:1},titles:["narchi - A neural network architecture definition package","API Reference","Command line tool","narchi - A neural network architecture definition package","Json Schema"],titleterms:{api:1,architectur:[0,3],argument:2,block:1,command:2,content:3,definit:[0,3],document:3,exampl:[0,3],featur:[0,3],graph:1,indic:3,instanti:1,json:4,line:2,load:2,main:[0,3],modul:1,name:2,narchi:[0,1,3],narchi_cli:2,network:[0,3],neural:[0,3],option:2,output:2,packag:[0,3],posit:2,propag:1,refer:1,relat:2,render:[1,2],reshape_spec:[],schema:[1,2,4],sympi:1,tabl:3,teaser:[0,3],tool:2,valid:2}})
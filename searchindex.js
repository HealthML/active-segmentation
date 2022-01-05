Search.setIndex({docnames:["active_learning","active_segmentation","brats","datasets","functional","index","inferencing","main","metric_tracking","models","modules","query_strategies"],envversion:{"sphinx.domains.c":2,"sphinx.domains.changeset":1,"sphinx.domains.citation":1,"sphinx.domains.cpp":4,"sphinx.domains.index":1,"sphinx.domains.javascript":2,"sphinx.domains.math":2,"sphinx.domains.python":3,"sphinx.domains.rst":2,"sphinx.domains.std":2,"sphinx.ext.intersphinx":1,"sphinx.ext.todo":2,"sphinx.ext.viewcode":1,sphinx:56},filenames:["active_learning.rst","active_segmentation.rst","brats.rst","datasets.rst","functional.rst","index.rst","inferencing.rst","main.rst","metric_tracking.rst","models.rst","modules.rst","query_strategies.rst"],objects:{"":{active_learning:[0,0,0,"-"],inferencing:[6,0,0,"-"],main:[7,0,0,"-"]},"active_learning.ActiveLearningPipeline":{run:[0,2,1,""]},"datasets.brats_data_module":{BraTSDataModule:[3,1,1,""]},"datasets.brats_data_module.BraTSDataModule":{discover_paths:[3,2,1,""],label_items:[3,2,1,""],train_dataloader:[3,2,1,""]},"datasets.data_module":{ActiveLearningDataModule:[3,1,1,""]},"datasets.data_module.ActiveLearningDataModule":{data_channels:[3,2,1,""],label_items:[3,2,1,""],setup:[3,2,1,""],test_dataloader:[3,2,1,""],test_set_size:[3,2,1,""],train_dataloader:[3,2,1,""],training_set_size:[3,2,1,""],unlabeled_dataloader:[3,2,1,""],unlabeled_set_size:[3,2,1,""],val_dataloader:[3,2,1,""],validation_set_size:[3,2,1,""]},"datasets.pascal_voc_data_module":{PILMaskToTensor:[3,1,1,""],PascalVOCDataModule:[3,1,1,""]},"datasets.pascal_voc_data_module.PascalVOCDataModule":{label_items:[3,2,1,""]},"functional.losses":{BCEDiceLoss:[4,1,1,""],BCELoss:[4,1,1,""],DiceLoss:[4,1,1,""],FalsePositiveDiceLoss:[4,1,1,""],FalsePositiveLoss:[4,1,1,""],SegmentationLoss:[4,1,1,""]},"functional.losses.BCEDiceLoss":{forward:[4,2,1,""],training:[4,3,1,""]},"functional.losses.BCELoss":{forward:[4,2,1,""],training:[4,3,1,""]},"functional.losses.DiceLoss":{forward:[4,2,1,""],training:[4,3,1,""]},"functional.losses.FalsePositiveDiceLoss":{forward:[4,2,1,""],training:[4,3,1,""]},"functional.losses.FalsePositiveLoss":{forward:[4,2,1,""],training:[4,3,1,""]},"functional.losses.SegmentationLoss":{flatten_tensor:[4,2,1,""],training:[4,3,1,""]},"functional.metrics":{DiceScore:[4,1,1,""],HausdorffDistance:[4,1,1,""],Sensitivity:[4,1,1,""],Specificity:[4,1,1,""],dice_score:[4,4,1,""],hausdorff_distance:[4,4,1,""],sensitivity:[4,4,1,""],specificity:[4,4,1,""]},"functional.metrics.DiceScore":{compute:[4,2,1,""],update:[4,2,1,""]},"functional.metrics.HausdorffDistance":{compute:[4,2,1,""],update:[4,2,1,""]},"functional.metrics.Sensitivity":{compute:[4,2,1,""],update:[4,2,1,""]},"functional.metrics.Specificity":{compute:[4,2,1,""],update:[4,2,1,""]},"inferencing.Inferencer":{inference:[6,2,1,""]},"metric_tracking.combined_per_epoch_metric":{CombinedPerEpochMetric:[8,1,1,""]},"metric_tracking.combined_per_epoch_metric.CombinedPerEpochMetric":{compute:[8,2,1,""],reset:[8,2,1,""],update:[8,2,1,""]},"metric_tracking.combined_per_image_metric":{CombinedPerImageMetric:[8,1,1,""]},"metric_tracking.combined_per_image_metric.CombinedPerImageMetric":{compute:[8,2,1,""],reset:[8,2,1,""],update:[8,2,1,""]},"models.pytorch_fcn_resnet50":{PytorchFCNResnet50:[9,1,1,""]},"models.pytorch_fcn_resnet50.PytorchFCNResnet50":{eval:[9,2,1,""],forward:[9,2,1,""],input_dimensionality:[9,2,1,""],parameters:[9,2,1,""],test_step:[9,2,1,""],train:[9,2,1,""],training:[9,3,1,""],training_step:[9,2,1,""],validation_step:[9,2,1,""]},"models.pytorch_model":{PytorchModel:[9,1,1,""]},"models.pytorch_model.PytorchModel":{configure_loss:[9,2,1,""],configure_optimizers:[9,2,1,""],get_test_metrics:[9,2,1,""],get_train_metrics:[9,2,1,""],get_val_metrics:[9,2,1,""],input_dimensionality:[9,2,1,""],predict:[9,2,1,""],predict_step:[9,2,1,""],setup:[9,2,1,""],test_epoch_end:[9,2,1,""],test_step:[9,2,1,""],training:[9,3,1,""],training_epoch_end:[9,2,1,""],training_step:[9,2,1,""],validation_epoch_end:[9,2,1,""],validation_step:[9,2,1,""]},"models.pytorch_u_net":{PytorchUNet:[9,1,1,""]},"models.pytorch_u_net.PytorchUNet":{eval:[9,2,1,""],forward:[9,2,1,""],input_dimensionality:[9,2,1,""],parameters:[9,2,1,""],precision:[9,3,1,""],predict_step:[9,2,1,""],test_step:[9,2,1,""],train:[9,2,1,""],training:[9,3,1,""],training_step:[9,2,1,""],use_amp:[9,3,1,""],validation_step:[9,2,1,""]},"models.u_net":{UNet:[9,1,1,""]},"models.u_net.UNet":{forward:[9,2,1,""],training:[9,3,1,""]},"query_strategies.query_strategy":{QueryStrategy:[11,1,1,""]},"query_strategies.query_strategy.QueryStrategy":{select_items_to_label:[11,2,1,""]},active_learning:{ActiveLearningPipeline:[0,1,1,""]},datasets:{brats_data_module:[3,0,0,"-"],data_module:[3,0,0,"-"],pascal_voc_data_module:[3,0,0,"-"]},functional:{losses:[4,0,0,"-"],metrics:[4,0,0,"-"]},inferencing:{Inferencer:[6,1,1,""]},main:{run_active_learning_pipeline:[7,4,1,""],run_active_learning_pipeline_from_config:[7,4,1,""]},metric_tracking:{combined_per_epoch_metric:[8,0,0,"-"],combined_per_image_metric:[8,0,0,"-"]},models:{pytorch_fcn_resnet50:[9,0,0,"-"],pytorch_model:[9,0,0,"-"],pytorch_u_net:[9,0,0,"-"],u_net:[9,0,0,"-"]},query_strategies:{query_strategy:[11,0,0,"-"]}},objnames:{"0":["py","module","Python module"],"1":["py","class","Python class"],"2":["py","method","Python method"],"3":["py","attribute","Python attribute"],"4":["py","function","Python function"]},objtypes:{"0":"py:module","1":"py:class","2":"py:method","3":"py:attribute","4":"py:function"},terms:{"0":[3,4,7,9],"0001":[7,9],"03385":9,"04597":9,"1":[3,4,7,9],"10":2,"12":2,"120":2,"1249":4,"1505":9,"1512":9,"155":2,"16":7,"19":2,"2":[3,4,9],"20":[2,3],"2012":[2,3],"2013":2,"2016":4,"2018":2,"24":2,"240":2,"245":2,"25":9,"26":2,"285":2,"2d":[3,4],"3":[3,4,9],"32":9,"3d":[3,4,8,9],"3t":2,"4":[2,7,9],"40":2,"405":2,"471":2,"5":9,"50":[7,9],"6":9,"60":2,"66":2,"7":9,"75":9,"8":9,"80":2,"9":9,"95":4,"abstract":[3,9],"class":[0,3,4,6,8,9,11],"default":[0,2,3,7,8,9],"do":2,"final":8,"float":[4,7,8,9],"function":[1,7,8,10],"int":[0,3,4,7,8,9],"new":[0,8],"return":[3,4,7,8,9,11],"static":[3,4,9],"true":[0,9],"while":[7,9],A:[0,4,6,8,9],As:4,By:2,For:4,If:[4,7,8],In:[2,4,8],It:2,Of:2,The:[0,3,4,5,6,7,8,9],To:2,_:8,_aggregated_:8,abc:[4,9],abl:0,about:9,ac:3,accord:2,accordingli:8,acquir:2,across:8,activ:[0,2,3,7],active_learn:[1,10],active_learning_framework:5,activelearningdatamodul:[0,3],activelearningpipelin:0,adam:9,adapt:[4,9],addit:11,addition:8,aeroplan:3,affect:2,aggreg:[4,8],ahmad:4,ahmadi:4,algorithm:[0,7,9],all:[2,4,8,9],all_image_loc:4,also:8,amount:[3,6],an:[0,4,7,8],ani:7,anim:3,annot:[2,3],annual:2,anymor:[0,7],appli:4,applic:2,ar:[0,2,4,7,8],arbitrari:4,architectur:[0,7,9],arg:3,arxiv:9,asia:2,assum:8,avail:8,b:7,background:[4,9],base:[0,3,4,6,7,8,9,11],batch:[3,4,8,9],batch_idx:9,batch_siz:[3,7],bce:[4,9],bcediceloss:4,bceloss:4,been:2,befor:9,begin:9,behaviour:4,being:4,better:2,between:[3,4],bicycl:3,binari:4,bird:3,blob:9,block:[7,9],board:2,boat:3,bool:[0,3,4,7],bottl:3,bound:3,box:3,brain:9,brat:[3,5,6,7],brats17_2013_24_1:2,brats17_2013_28_1:2,brats17_2013_3_1:2,brats17_2013_5_1:2,brats17_cbica_apr_1:2,brats17_cbica_aqg_1:2,brats17_cbica_aqv_1:2,brats17_cbica_asa_1:2,brats17_cbica_ase_1:2,brats17_cbica_asy_1:2,brats17_cbica_awi_1:2,brats17_cbica_azd_1:2,brats17_cbica_bhb_1:2,brats17_tcia_101_1:2,brats17_tcia_141_1:2,brats17_tcia_149_1:2,brats17_tcia_151_1:2,brats17_tcia_152_1:2,brats17_tcia_175_1:2,brats17_tcia_186_1:2,brats17_tcia_202_1:2,brats17_tcia_242_1:2,brats17_tcia_266_1:2,brats17_tcia_276_1:2,brats17_tcia_282_1:2,brats17_tcia_298_1:2,brats17_tcia_321_1:2,brats17_tcia_343_1:2,brats17_tcia_387_1:2,brats17_tcia_420_1:2,brats17_tcia_430_1:2,brats17_tcia_449_1:2,brats17_tcia_469_1:2,brats17_tcia_498_1:2,brats17_tcia_607_1:2,brats17_tcia_618_1:2,brats17_tcia_621_1:2,brats17_tcia_624_1:2,brats17_tcia_639_1:2,brats17_tcia_640_1:2,brats18_2013_16_1:2,brats18_2013_19_1:2,brats18_2013_26_1:2,brats18_2013_9_1:2,brats18_cbica_abb_1:2,brats18_cbica_abm_1:2,brats18_cbica_abn_1:2,brats18_cbica_alx_1:2,brats18_cbica_aqg_1:2,brats18_cbica_aqj_1:2,brats18_cbica_aqq_1:2,brats18_cbica_aqy_1:2,brats18_cbica_arw_1:2,brats18_cbica_auq_1:2,brats18_cbica_aya_1:2,brats18_cbica_ayu_1:2,brats18_tcia01_221_1:2,brats18_tcia01_231_1:2,brats18_tcia01_429_1:2,brats18_tcia02_198_1:2,brats18_tcia02_300_1:2,brats18_tcia02_309_1:2,brats18_tcia02_314_1:2,brats18_tcia02_370_1:2,brats18_tcia02_377_1:2,brats18_tcia02_473_1:2,brats18_tcia03_133_1:2,brats18_tcia06_211_1:2,brats18_tcia06_603_1:2,brats18_tcia08_162_1:2,brats18_tcia08_167_1:2,brats18_tcia08_205_1:2,brats18_tcia09_255_1:2,brats18_tcia09_402_1:2,brats18_tcia09_620_1:2,brats18_tcia10_282_1:2,brats18_tcia10_387_1:2,brats18_tcia10_449_1:2,brats18_tcia10_629_1:2,brats_data_modul:3,bratsdatamodul:[1,10],bratsdataset:[1,10],bu:3,c:4,cache_s:3,calcul:[4,8,9],call:9,can:[3,4,8],car:3,cat:3,categori:[2,3],cbica:2,cdot:4,certifi:2,chair:3,challeng:2,channel:[3,4,9],checkpoint:[0,7],checkpoint_dir:[0,7],cli:7,clinic:2,coeffici:4,com:[4,9],combin:4,combined_per_epoch_metr:[1,10],combined_per_image_metr:[1,10],combinedperepochmetr:8,combinedperimagemetr:8,common:2,compos:2,comput:[4,8,9],confid:[8,9],confidence_level:8,config:7,config_file_nam:7,configur:[7,9],configure_loss:9,configure_optim:9,confirm:2,consid:4,consist:2,contain:[0,3,4,6,8],contributor:2,convolut:4,core:[2,3,9],correspond:2,cosineannealinglr:[0,7,9],cours:2,cow:3,creat:3,creation:3,cross:4,current:[3,9,11],custom:4,data:[0,3,5,6,7,8,11],data_channel:3,data_dir:[3,6,7],data_modul:[0,3],dataload:[3,9,11],dataloader_id:9,dataloader_idx:9,datamodul:[1,10],dataset:[1,5,6,7,8,9,10,11],dataset_config:[6,7],datset:3,decid:2,decod:[7,9],defin:[0,3,9],desir:7,detail:[5,9],diagnosi:2,diagon:4,dice:[4,8,9],dice_scor:[4,9],diceloss:4,dicescor:4,dict:[7,8],dictionari:[7,8],differ:[2,4,7,8],differenti:4,dim:[3,9],dimens:[4,8],dimension:[4,9],dine:3,dir_path:3,directori:[0,3,6,7],disabl:[0,7],discov:3,discover_path:3,discuss:4,distanc:[4,8],divid:4,document:9,dog:3,doubl:9,down:9,dsc:4,dure:[0,7,9],dynam:[0,7,9],e:[0,6,7,8,9],each:[3,4,7,8,9],earli:[0,7],early_stop:[0,7],edema:2,either:[4,8,9],element:4,en:[4,9],enabl:[0,7],encod:[7,9],end:9,enhanc:2,ensur:4,entropi:4,environ:0,epoch:[0,3,7,8,9],equal:9,establish:2,euclidean:4,europ:2,eval:9,evalu:[4,9],exampl:[7,8],except:4,exclud:[4,9],execut:[7,9],expect:8,experi:[0,2,7],experiment_nam:7,experiment_tag:7,expert:2,factor:4,fals:[0,4,7],falsepositivediceloss:4,falsepositiveloss:4,fausto:4,fcn_resnet50:9,featur:9,file:[2,3,7],fill:2,first:[4,9],fit:[0,9],five:2,flag:[3,7],flair:3,flatten:4,flatten_tensor:4,flexibl:9,fluid:2,fn:4,follow:2,form:8,formul:4,forward:[4,9],fp:[4,9],frac:4,framework:[4,9],from:[2,3,4,7,8,9],full:7,fulli:4,further:[2,3,9],g:[0,6,7,8,9],gener:[5,6,9],get:9,get_test_metr:9,get_train_metr:9,get_val_metr:9,github:[4,9],given:[3,4,6,8,9],glioblastoma:2,glioma:2,global:8,gpu:[0,7],grade:2,ground:2,group:2,gz:3,ha:3,hausdorf:4,hausdorff95:[8,9],hausdorff:[4,8],hausdorff_dist:4,hausdorffdist:4,have:[2,4,8],healthi:2,height:[4,8],held:2,holdout:2,hook:9,hors:3,host:3,household:3,how:4,hp_optimis:7,html:[4,9],http:[3,4,9],hyperparamet:7,id:[3,8,11],ignor:8,imag:[3,4,8,9],image_id:8,implement:[4,9],improv:11,in_channel:9,includ:[2,3,4],inclus:2,index:[5,9],infer:6,inferenc:[1,10],inform:[3,5],init_featur:9,initi:3,input:[4,9],input_dimension:9,institut:2,instruct:4,integ:8,intern:8,introduction_guid:9,io:[4,9],issu:4,item:[3,11],iter:[7,8,9],its:4,keep:3,kei:8,kera:3,kwarg:[3,9,11],label:[0,2,3,7,11],label_item:3,lambda:4,laplacian:4,latest:[4,9],layer:[4,8],learn:[0,3,7,9],learning_r:[7,9],level:[3,7,8,9],lightn:[0,9],lightningdatamodul:3,lightningmodul:9,likelihood:4,list:[2,3,8,9],load:3,locat:4,log:[4,9],logger:0,logic:6,loop:9,loss:[0,1,7,9,10],lower:2,lr_schedul:[0,7,9],mai:8,main:[6,10],manag:3,map:8,mask:[4,9],master:9,match:9,mateuszbuda:9,max:8,maximum:4,mean:[4,8],measur:9,medic:4,memori:3,method:[4,8,9],metric:[1,8,9,10],metric_track:[1,10],metrics_to_aggreg:8,milletari:4,min:8,minimum:[7,9],modal:3,mode:9,model:[0,1,4,6,7,10,11],model_config:7,model_selection_criterion:[0,7],modul:[0,3,4,5,6,8,9,10],monitor:3,motorbik:3,move:[2,3,7,9],mri:2,mrt:8,multimod:2,multimodalbraintumorsegment:2,multipl:[3,4,8,9],must:[4,8],mutual:2,n:[4,8],name:[2,6,7,8,9],nassir:4,navab:4,necrot:2,neg:4,net:[4,7,9],network:4,neural:4,neuroradiologist:2,next:11,nii:3,nn:[4,9],non:2,none:[0,3,4,6,7,8,9],normal:4,note:4,num_level:[7,9],num_work:[3,7],number:[0,3,4,7,8,9,11],number_of_item:11,object:[0,3,6,9,11],one:[3,4,7,8,9],onli:9,optim:9,optin:9,option:[0,3,4,7,8,9],org:9,origin:[4,9],other:3,otherwis:[4,8],our:2,out:2,out_channel:9,output:[4,8,9],over:4,overwritten:3,ox:3,packag:4,page:[4,5],param:[3,6,7,9,11],paramet:[0,3,4,7,8,9,11],pascal:3,pascal_voc_data_modul:3,pascalvocdatamodul:[1,10],pass:[4,7,8,9],path:[3,7],patholog:2,patient:2,pdf:9,per:[4,8],percentil:4,perform:9,person:3,pilmasktotensor:3,pin_memori:3,pipelin:[0,7],pixel:[3,4],plant:3,point:4,posit:4,pot:3,pre:[2,4],precis:9,predict:[4,6,7,8,9],predict_step:9,prediction_count:[6,7],prediction_dir:[6,7],process:2,produc:9,project:7,proper:2,protocol:2,provid:[0,2,8],publish:2,py:9,pytorch:[0,3,4,9,11],pytorch_fcn_resnet50:9,pytorch_lightn:[3,9],pytorch_model:9,pytorch_u_net:9,pytorchfcnresnet50:[1,10],pytorchmodel:[0,1,10],pytorchunet:[1,10],queri:[0,7],query_strategi:[1,10],querystrategi:[0,1,10],random:3,random_sampl:3,randomli:2,rate:[0,7,9],re:2,readi:8,readthedoc:[4,9],receiv:2,recurs:9,reducelronplateau:[0,7,9],reduct:[4,8],region:2,regist:4,relat:8,repres:[2,3,4,11],requir:9,reset:8,resnet50:9,resnet:9,resolut:2,robot:3,run:[0,6,7],run_active_learning_pipelin:7,run_active_learning_pipeline_from_config:7,s:[2,4,9],same:[4,8],sampl:[3,5,9],save:[0,7],scalar:4,scan:[2,3,8],scanner:2,scatter:[4,8],schedul:9,scipi:4,score:8,search:5,see:[4,9],segment:[3,4,7,8,9],segmentationloss:4,sei:4,select:[2,3,11],select_items_to_label:11,sensit:[4,8,9],separ:8,sequenc:3,set:[2,3,7,9],setup:[2,3,9],sever:[2,8],sgd:9,shape:[4,8],share:2,sharp:4,sheep:3,should:[0,3,4,9,11],shuffl:3,sigmoid:[4,8],similar:4,simul:[0,7],sinc:2,singl:[4,8],size:[3,4,5,7,8,9],skull:2,slice:[3,4,8],slices_per_imag:[4,8],smooth:4,so:2,sofa:3,solid:2,sourc:[0,3,4,6,7,8,9,11],space:4,specif:[3,4,7,8,9,11],specifi:9,speed:[3,4],split:5,spot:2,stage:[3,9],start:[7,9],starter:9,state:8,step:[2,7,9],stop:[0,7],store:7,str:[0,4,7,8,9],strategi:[0,7,11],string:[0,7,8,9],strip:2,structur:[2,3],subclass:3,subject:2,subpackag:5,subset:[8,11],sum:4,sweep:7,tabl:3,tag:7,take:8,taken:[2,9],target:[4,8],task:[2,4,8,9],tbd:3,tcia:2,tensor:[4,8,9],test:[3,9],test_dataload:3,test_epoch_end:9,test_metr:9,test_metric_confidence_level:9,test_set_s:3,test_step:9,them:8,therefor:4,thi:[3,4,7,8,9],three:2,thu:4,tn:4,torch:[4,9],torchmetr:[4,8],torchvis:9,total:2,toward:[7,9],tp:4,track:8,train:[0,2,3,4,7,9],train_dataload:3,train_metr:9,train_metric_confidence_level:9,trainer:9,training_epoch_end:9,training_set_s:3,training_step:9,truth:2,tumor:2,tupl:3,tv:3,two:4,type:[3,4,8,9],typic:4,u:[2,7,9],u_net:[7,9],uk:3,um_work:3,unet:[1,10],uniform:4,union:8,unlabel:[2,3,11],unlabeled_dataload:3,unlabeled_set_s:3,up:[3,4],updat:[0,4,7,8,9],us:[0,4,6,7,8,9],use_amp:9,usual:9,v:4,val:9,val_dataload:3,valid:[2,3,9],validation_epoch_end:9,validation_set_s:3,validation_step:9,valu:[4,8,9],vehicl:3,version:4,view:4,vision:9,visual:3,voc:3,volumetr:4,w:7,wa:[2,4,9],wandb_project_nam:7,we:2,weight:9,well:2,were:[2,4],when:[0,7],where:[0,4,7,8],whether:4,which:[4,7,8,9],who:2,whole:[2,3,8],whose:[4,8],width:[4,8],worker:[3,7],wrap:9,wrapper:4,x:[4,9],y:4,your:9},titles:["active_learning","active_learning_framework","The BraTS Dataset","datasets","functional","Welcome to Active Segmentation\u2019s documentation!","inferencing","main module","metric_tracking","models","src","query_strategies"],titleterms:{"function":4,The:2,activ:5,active_learn:0,active_learning_framework:1,brat:2,bratsdatamodul:3,bratsdataset:3,combined_per_epoch_metr:8,combined_per_image_metr:8,content:5,data:2,datamodul:3,dataset:[2,3],detail:2,document:5,gener:2,indic:5,inferenc:6,inform:2,loss:4,main:7,metric:4,metric_track:8,model:9,modul:7,pascalvocdatamodul:3,pytorchfcnresnet50:9,pytorchmodel:9,pytorchunet:9,query_strategi:11,querystrategi:11,s:5,sampl:2,segment:5,size:2,split:2,src:10,subpackag:1,tabl:5,unet:9,welcom:5}})
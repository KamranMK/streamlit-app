---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
Cell In[7], line 3
      1 # Create the AdalComponent - this should NOT hang now
      2 logger.info("Creating AdalComponent (using OfflineGenerator)...")
----> 3 adal_component = EmailClassificationOptimizer(
      4     model_client=model_client,
      5     model_kwargs=model_config,
      6     backward_engine_model_config=model_config,
      7     teacher_model_config=model_config,
      8     text_optimizer_model_config=model_config,
      9 )
     10 logger.info("AdalComponent created successfully!")

File ~/adam-email-routing/prompt-optimizer/optimizer/optimizer.py:41, in EmailClassificationOptimizer.__init__(self, model_client, model_kwargs, backward_engine_model_config, teacher_model_config, text_optimizer_model_config)
     38 eval_fn = AnswerMatchAcc(type="exact_match").compute_single_item
     40 # Create loss function with explicit parameters to prevent automatic initialization
---> 41 loss_fn = EvalFnToTextLoss(
     42     eval_fn=eval_fn,
     43     eval_fn_desc="Exact match between predicted and ground truth category",
     44     backward_engine=None,
     45     model_client=model_client,
     46     model_kwargs=backward_engine_model_config
     47 )
     49 # Store configs for later use
     50 self.backward_engine_model_config = backward_engine_model_config

File ~/.conda/envs/adalflow/lib/python3.10/site-packages/adalflow/optim/text_grad/text_loss_with_eval_fn.py:85, in EvalFnToTextLoss.__init__(self, eval_fn, eval_fn_desc, backward_engine, model_client, model_kwargs)
     80     log.info(
     81         "EvalFnToTextLoss: No backward engine provided. Creating one using model_client and model_kwargs."
     82     )
     83     if model_client and model_kwargs:
---> 85         self.set_backward_engine(backward_engine, model_client, model_kwargs)
     86 else:
     87     if not isinstance(backward_engine, BackwardEngine):

File ~/.conda/envs/adalflow/lib/python3.10/site-packages/adalflow/optim/text_grad/text_loss_with_eval_fn.py:175, in EvalFnToTextLoss.set_backward_engine(self, backward_engine, model_client, model_kwargs)
    171 if not backward_engine:
    172     log.info(
    173         "EvalFnToTextLoss: No backward engine provided. Creating one using model_client and model_kwargs."
    174     )
--> 175     self.backward_engine = BackwardEngine(model_client, model_kwargs)
    176 else:
    177     if type(backward_engine) is not BackwardEngine:

TypeError: BackwardEngine.__init__() takes 1 positional argument but 3 were given

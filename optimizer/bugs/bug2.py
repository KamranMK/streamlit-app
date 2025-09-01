---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
Cell In[8], line 3
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

File ~/adam-email-routing/prompt-optimizer/optimizer/optimizer.py:54, in EmailClassificationOptimizer.__init__(self, model_client, model_kwargs, backward_engine_model_config, teacher_model_config, text_optimizer_model_config)
     52 self.teacher_model_config = teacher_model_config
     53 self.text_optimizer_model_config = text_optimizer_model_config
---> 54 self._main_model_client = model_client  # Store for later backward engine setup
     56 # Initialize parent WITHOUT triggering backward engine creation
     57 super().__init__(
     58     task=task,
     59     eval_fn=eval_fn,
   (...)
     65     text_optimizer_model_config=None,
     66 )

File ~/.conda/envs/adalflow/lib/python3.10/site-packages/adalflow/core/component.py:877, in Component.__setattr__(self, name, value)
    875 if isinstance(value, Component):
    876     if components is None:
--> 877         raise AttributeError(
    878             "cant assign component before Component.__init__() call"
    879         )
    880     remove_from(self.__dict__)
    881     components[name] = value

AttributeError: cant assign component before Component.__init__() call

raw_shots: 0, bootstrap_shots: 1
No demo parameters found.
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
Cell In[17], line 1
----> 1 trainer.fit(train_dataset=train_samples, val_dataset=val_samples, test_dataset=test_samples)

File ~/.conda/envs/adalflow/lib/python3.10/site-packages/adalflow/optim/trainer/trainer.py:504, in Trainer.fit(self, adaltask, train_loader, train_dataset, val_dataset, test_dataset, debug, save_traces, raw_shots, bootstrap_shots, resume_from_ckpt, backward_pass_setup)
    500         raise ValueError(
    501             "train_dataset should not be tuple, please use dict or a dataclass or with DataClass"
    502         )
    503 #  prepare optimizers
--> 504 self.optimizers: List[Optimizer] = self.adaltask.configure_optimizers(
    505     **self.text_optimizers_config_kwargs
    506 )
    507 self.text_optimizers = [
    508     opt for opt in self.optimizers if isinstance(opt, TextOptimizer)
    509 ]
    510 self.demo_optimizers = [
    511     opt for opt in self.optimizers if isinstance(opt, DemoOptimizer)
    512 ]

File ~/.conda/envs/adalflow/lib/python3.10/site-packages/adalflow/optim/trainer/adal.py:183, in AdalComponent.configure_optimizers(self, *args, **text_optimizer_kwargs)
    181     raise ValueError("Text optimizer model config is not configured.")
    182 if not self.text_optimizer_model_config.get("model_client"):
--> 183     raise ValueError("Model client is not configured.")
    184 if not self.text_optimizer_model_config.get("model_kwargs"):
    185     raise ValueError("Model kwargs is not configured.")

ValueError: Model client is not configured.

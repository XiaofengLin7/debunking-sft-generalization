from ragen.llm_agent.generation import LLMGenerationManager
from dataclasses import dataclass
from verl.utils.tracking import Tracking
import torch
from typing import List, Any, Tuple, Dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto

@dataclass
class GenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int 
    max_response_length: int
    max_obs_length: int
    logging: dict
    num_gpus: int
    no_think_rl: bool=False
    append_obs: bool=False


class ReilGenerationManager(LLMGenerationManager):
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        env_class,
        config: GenerationConfig,
        logger: Tracking,
        is_validation: bool = False,
    ):
        super().__init__(
            tokenizer,
            actor_rollout_wg,
            env_class,
            config,
            logger,
            is_validation,
        )

    def _postprocess_responses(self, responses: torch.Tensor, envs: List[Any]) -> torch.Tensor:
        """Process responses to remove 1. multiple answers or 2. reward hacking attempts."""
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )

        responses_str = self._process_answer_tag(responses_str)
        
        if self.config.no_think_rl:
            # if no_think_rl is enabled, only keep action in the str
            actions, _ = self.env_class.postprocess_predictions(envs, responses_str)
            responses_str=[f"<answer>{envs[idx].ACTION_LOOKUP[action]}</answer>" for idx, action in enumerate(actions)]
            print("RESPONSES:", responses_str)
        responses = self._batch_tokenize(responses_str)
        return responses, responses_str

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        """Process next observations from environment."""
        next_obs_ids = self.tokenizer(
            next_obs, 
            padding='longest',
            return_tensors='pt'
        )['input_ids']
        
        if next_obs_ids.shape[1] > self.config.max_obs_length:
            print("[WARNING] OBSERVATION TOO LONG, CONSIDER CHANGING YOUR CONFIG")
            next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]
            
        return next_obs_ids
    
    def _update_rolling_state(self, rollings, cur_responses: torch.Tensor, 
                            next_obs_ids: torch.Tensor) -> Dict:
        """Update rolling state with new responses and observations."""
        if self.config.append_obs:
            # Concatenate and handle padding
            new_input_ids = self.tensor_fn.concatenate_with_padding([
                rollings.batch['input_ids'],
                cur_responses,
                next_obs_ids
            ])
        else:
            new_input_ids = self.tensor_fn.concatenate_with_padding([
                next_obs_ids
            ])
        
        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })
    
    def run_llm_loop(self, gen_batch, envs: List[Any],
                    initial_input_ids: torch.Tensor,
                    output_dir: str,
                    global_steps: int) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""
        # Setup visualization and Initialize states
        trajectory = self._setup_visualization()
        
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {'responses': initial_input_ids[:, []]}
        
        active_mask = torch.ones(gen_batch.batch['input_ids'].shape[0], dtype=torch.bool)
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch


        # Main generation loop
        for step in range(self.config.max_turns):
            if not active_mask.sum():
                break
            # rollings.batch = self.tensor_fn.cut_to_effective_len(
            #     rollings.batch,
            #     keys=['input_ids', 'attention_mask', 'position_ids']
            # )
            breakpoint()
            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            }, meta_info=gen_batch.meta_info)

            rollings_active, pad_size = pad_dataproto_to_divisor(rollings_active, self.actor_rollout_wg.world_size)
            gen_output = self.actor_rollout_wg.generate_sequences(rollings_active)
            gen_output = unpad_dataproto(gen_output, pad_size=pad_size)

            meta_info = gen_output.meta_info            
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'],envs=envs)
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # Update visualization
            self._update_trajectory(trajectory, envs, responses_str, active_mask)

            # Execute in environment and process observations
            next_obs, dones = self.env_class.execute_predictions(
                envs, responses_str, responses_ids, self.tokenizer
            )
            active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_num_list.append(active_mask.sum().item())
            next_obs_ids = self._process_next_obs(next_obs)
            
            # Update states
            rollings = self._update_rolling_state(
                rollings,
                responses_ids,
                next_obs_ids
            )
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                next_obs_ids
            )
        # print("ACTIVE_TRAJ_NUM:", active_num_list)
        
        # Save trajectory and return final output
        self._save_trajectory(trajectory, output_dir, global_steps)
        return self._compose_final_output(original_left_side, original_right_side, meta_info)
    

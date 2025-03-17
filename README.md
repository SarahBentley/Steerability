
# Measuring the Steerability of Generative Models

Source code for the paper "What’s Producible May Not Be Reachable:
Measuring the Steerability of Generative Models" by Keyon Vafa, Sarah Bentley, Jon Kleinberg, and Sendhil Mullainathan.

### Installing Dependencies

To install the dependencies, run the following command:
```bash
pip install -r requirements.txt
```

## The Steerability of Image Generation Models
The data and analysis from our benchmark on the steerability of image generation models and the results of our RL steering method are in the `image_generation` directory.

### Data
We provide the following data:
-  `steering_all_seeds.csv`: Our primary survey on steering image generation models. Each row contains a user's steering attempt, and the data is organized by users. The `dreamsim` and `clip_similarity` columns contain similarity scores between the goal_image and generated_image.
- `steering_x_seeds.csv`: Users perform steering as usual, but they are constrained to x different seeds, which are assigned to different attempts at random.
- `pom1.csv`: Prompt-Output-Misalignment data for first attempts. The `choice` column contains the image chosen by the user (either the goal or first attempt image), while `goal_chosen` is a binary variable indicating whether the user chose the goal image.
- `pom5.csv`: Prompt-Output-Misalignment data for fifth attempts. The `choice` column contains the image chosen by the user (either the goal or fifth attempt image), while `goal_chosen` is a binary variable indicating whether the user chose the goal image.
- `improvement.csv`: Users' assessments of whether first attempt or last attempt images are closer to their corresponding goal images. `last_chosen` is a binary variable indicating whether the user chose the last attempt image.
- `sat_rating_10.csv`: User satisfaction ratings of the generated images, in comparison to the goal images, on a 10-point scale.
- `sat_rating_4.csv`: User satisfaction ratings of the generated images, in comparison to the goal images, on a 4-point scale.
-  `blind_steering_x.csv`: The results of blind steering, when an LLM steers an image generation model by producing x variations of a user's first attempt prompt. The `score` column contains the DreamSim score of the LLM's generated image, and the `user_score` column contains the DreamSim score of the user's first attempt image.
- `choosing_seed.csv`: Users perform steering while choosing a seed each time they generate an image.
- `img_steering.csv`: Users steer image generation models by iteratively choosing from variations of their current generated image.
- `img_steering_rl.csv`: We leverage a reinforcement learning technique to optimally provide image variations.
- `img_steering_rl_tiles.csv`: Image steering with RL on images of tiles.
- `bookshelf_steering.csv`: Steering towards one specific goal image -- an image of books -- for our primary figure.
- `summary_metrics.csv`: A summary of our main metrics -- improvement rating, POM1, POM5, and more -- for each model.

### Analysis
The `calculate_summary_metrics.py` script can be used to generate the metrics for our benchmark on steerability reported in the paper. The notebook `tables_and_figures.ipynb` can be used to explore the data and generate the tables and figures in our paper.


## The Steerability of Large Language Models
The data and analysis from our benchmark on the steerability of large language models can be found in the `LLMs` directory.

## Data
We provide the following data:
-  `steering_llms.csv`: Our primary survey on steering large language models. Each row contains a user's steering attempt, and the data is organized by users. The `secret_prompt` column contains the prompt we used to generate the `goal_headline` from the `original_headline`. The `user_prompt` column contains users' prompt attempts at generating the `goal_headline`.
- `improvement.csv`: Users' assessments of whether first attempt or last attempt generated headlines are closer to their corresponding goal headlines. `last_chosen` is a binary variable indicating whether the user chose the last attempt headline.
- `sat_ratings.csv`: User satisfaction ratings of the generated headlines, in comparison to the goal headlines, on a 4-point scale, with 1 being Very Unsatisfied and 4 being Very Satisfied.

## Analysis
The `calculate_summary_metrics.py` script can be used to calculate the metrics reported in the paper.

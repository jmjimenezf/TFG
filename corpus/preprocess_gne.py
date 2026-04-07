# Script to preprocess the GNE to map the labels to Ekman set of emotions 
# and filter the examples that do not belong to the that set.
# The GNE corpus has the following labels: "joy", "sadness", "anger", "fear", "disgust", "surprise", "neutral". 
# The Ekman set of emotions is: "joy", "sadness", "anger", "fear", "disgust", "surprise". 
# We will map the "neutral" label to "neutral" and filter out the examples that do not belong to the Ekman set of emotions. 
# The resulting corpus will be saved in a new file called gne_ekman.jsonl containing:
# id, headline, gold from ekman set.
import json

input_file = "gne-release-v1.0.jsonl"
output_file = "gne_ekman.jsonl"
log_file = "gne_preprocess.log"

# Map positive_surprise and negative_surprise to surprise, and neutral to neutral
# Emotions dropped: Annoyance, guilt, love, pessimism, optimism, pride, shame, trust
# annotations -> dominate_emotion -> gold
EKMAN_SET = {"joy", "sadness", "anger", "fear", "disgust", "surprise"}
count = 0
filtered = 0
with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
    for line in f_in:
        count += 1
        example = json.loads(line)
        emotion = example["annotations"]["dominant_emotion"]["gold"]
        if emotion == "positive_surprise":
            emotion = "surprise"
        elif emotion == "negative_surprise":
            emotion = "surprise"
        if emotion in EKMAN_SET:
            filtered += 1
            example["gold"] = emotion
            # Keep only id, headline and gold
            filtered_example = {
                "id": example["id"],
                "headline": example["headline"],
                "gold": example["gold"]
            }
            f_out.write(json.dumps(filtered_example) + "\n")
print(f"Total samples: {count}")
print(f"Filtered samples: {filtered}")
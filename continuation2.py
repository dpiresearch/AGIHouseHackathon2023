from transformers import AutoProcessor, MusicgenForConditionalGeneration
import numpy as np

# Load processor
processor = AutoProcessor.from_pretrained("facebook/musicgen-large")

# Load model locally ( previously saved )
model = MusicgenForConditionalGeneration.from_pretrained("./fb-musicgen-large.model")

bpmString = "at 120 beats per minute"           # beats per minute
styleString = "in the classical Chopin style"   # style
otherString = "with piano only"                 # other embellishments

# The original idea was to ask GPT-4 for a list of prompts for musicgen given a scenario
# This was the result, but eventually not used as the music slices were disjoint

city_scenes = [
    "Skyscrapers casting morning shadows",
    "Horns blaring distant traffic jams",
    "Street vendors calling out wares",
    "Pedestrians hustle cross busy intersections",
    "Subway rumbling beneath feet",
    "Taxi cabs weaving lanes",
    "Children laughing park playground",
    "Elevator music high-rise buildings",
    "Neon lights flicker nightlife",
    "Bicycles dodge through crowds",
    "Pigeons scatter from food crumbs",
    "Sirens wailing urgent emergencies",
    "Buskers strumming upbeat street tunes",
    "Coffee shops steam and chatter",
    "Raindrops patter on umbrellas",
    "Jazz club sultry evening vibes",
    "Billboards flashing bright advertisements",
    "Fountains splash in plazas",
    "High heels click on sidewalks",
    "News stands rustle papers",
    "Delivery trucks unload hurriedly",
    "Lunch crowds queue favorite spots",
    "Tunnels echo with acoustics",
    "City square hosts lively festivals",
    "Trams ding along routes",
    "Rooftop parties buzz nightlife",
    "Stadium roars from game excitement",
    "Distant ship horns harbor",
    "Night markets sizzle and aroma",
    "Clock tower chimes noon hour"
]

# Our initial sample is based on a text prompt
aa = 0
sampling_rate = model.config.audio_encoder.sampling_rate

scene = "Ocean timelapse"
promptStr = scene + " " + styleString + " " + bpmString + " " + otherString
print("processing: " + scene)
inputs = processor(
    text=[promptStr],
    padding=True,
    return_tensors="pt",
)

audio_values = model.generate(**inputs, max_new_tokens=256)
print(audio_values.shape)
concatAudio = audio_values[0,0].numpy()

# Subsequent samples are audio prompted on the previous samples
for i in range(1, 5):
    sample = concatAudio
    sample = sample[: len(sample) // 2]

    inputs = processor(
        audio=sample,
        sampling_rate=sampling_rate,
        text=[promptStr],
        padding=True,
        return_tensors="pt",
    )

    audio_values2 = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=256)
    print(audio_values2.shape)

    concatAudio = np.concatenate((concatAudio, audio_values2[0, 0].numpy()))

# Write the whole thing out to a wav file
import scipy
print("writing: " + promptStr + ".wav")
scipy.io.wavfile.write(promptStr + ".wav", rate=sampling_rate, data=concatAudio)



from transformers import AutoProcessor, MusicgenForConditionalGeneration

import numpy as np

processor = AutoProcessor.from_pretrained("facebook/musicgen-large")
#AutoProcessor.save_pretrained(processor, "./fb-musicgen-large.proc")
#print("Saving processor")
#model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-large")
model = MusicgenForConditionalGeneration.from_pretrained("./fb-musicgen-large.model")
print("Loading model")
#MusicgenForConditionalGeneration.save_pretrained(model, "./fb-musicgen-large.model")
#print("Saving model")

bpmString = "at 90 beats per minute"
styleString = "in the style of 80s pop"
otherString = "with bassy drums and synth only"

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
short_scenes = [
    "Skyscrapers casting morning shadows",
    "Horns blaring distant traffic jams",
    "Street vendors calling out wares",
    "Clock tower chimes noon hour"
]

aa = 0
'''
for scene in city_scenes:
    if aa == 0:
        print("starting")
    else:
        print(aa)
    aa += 1
'''


for scene in short_scenes:
    promptStr = scene + " " + styleString + " " + bpmString + " " + otherString
    print("processing: " + scene)
    inputs = processor(
        text=[promptStr],
        padding=True,
        return_tensors="pt",
    )

    audio_values = model.generate(**inputs, max_new_tokens=512)

    import scipy

    print("writing: " + promptStr + ".wav")

    sampling_rate = model.config.audio_encoder.sampling_rate
    scipy.io.wavfile.write(promptStr + ".wav", rate=sampling_rate, data=audio_values[0, 0].numpy())
    if aa == 0:
        combined_data = audio_values[0, 0].numpy()
        overlap_data = audio_values[0, 0].numpy()
    else:
        combined_data = np.concatenate((combined_data, audio_values[0, 0].numpy()), axis=0)
        overlap_data = overlap_data + audio_values[0, 0].numpy()
    aa += 1
scipy.io.wavfile.write("all" + ".wav", rate=sampling_rate, data=combined_data)
scipy.io.wavfile.write("overlap" + ".wav", rate=sampling_rate, data=overlap_data)


#inputs = processor(
#    text=["80s pop track with bassy drums and synth", "90s rock song with loud guitars and heavy drums","2000 electronic harmonies"],
#    text=["2000 electronic harmonies"],
#    text=["10 bars of violin music like Jean Luc Ponty in F minor"],
#    text=["music for a timelapse video"],

#    padding=True,
#    return_tensors="pt",
#)

#audio_values = model.generate(**inputs, max_new_tokens=256)

#import scipy

#sampling_rate = model.config.audio_encoder.sampling_rate
#scipy.io.wavfile.write("Fminor.wav", rate=sampling_rate, data=audio_values[0, 0].numpy())


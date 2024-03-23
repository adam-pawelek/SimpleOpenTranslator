from lingua import Language, LanguageDetectorBuilder

detector = LanguageDetectorBuilder.from_languages(*Language.all()).build()
sentence =  """<2en> Lithuania! My homeland! you are like health: O quanto você deve ser valorizado, só ele vai descobrir, Kto cię stracił. Dziś piękność twą w całej ozdobie Widzę i opisuję, bo tęsknię po tobie. Panno święta, co Jasnej bronisz Częstochowy I w Ostrej świecisz Bramie! Ty, co gród zamkowy Nowogródzki ochraniasz z jego wiernym ludem! Jak mnie dziecko do zdrowia powróciłaś cudem (Gdy od płaczącej matki, pod Twoją opiekę Ofiarowany, martwą podniosłem powiekę;"""

for result in detector.detect_multiple_languages_of(sentence):
    print(f"{result.language.name}: '{sentence[result.start_index:result.end_index]}'")

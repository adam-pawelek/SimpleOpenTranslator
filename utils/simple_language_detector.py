from lingua import Language, LanguageDetectorBuilder



def one_language_auto_detect(text):
    detector = LanguageDetectorBuilder.from_all_languages().with_preloaded_language_models().build()
    #print(detector.detect_language_of("To obtain the confidence level along with the detected language, you can use the"))

    confidence_values = detector.compute_language_confidence_values(text)

    return  confidence_values[0].language, confidence_values[0].value


print(one_language_auto_detect("Hola hola no co ty tu robisz"))
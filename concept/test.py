from lingua import Language, LanguageDetectorBuilder
languages = [Language.ENGLISH, Language.FRENCH, Language.GERMAN, Language.SPANISH, Language.POLISH]
detector = LanguageDetectorBuilder.from_all_languages().with_preloaded_language_models().build()
print(detector.detect_language_of("To obtain the confidence level along with the detected language, you can use the" ))

confidence_values = detector.compute_language_confidence_values("To obtain the confidence level along with the detected language, you can use the ")


print(confidence_values[0].value)
print(confidence_values[0].language)






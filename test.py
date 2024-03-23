from lingua import Language, LanguageDetectorBuilder
languages = [Language.ENGLISH, Language.FRENCH, Language.GERMAN, Language.SPANISH, Language.POLISH]
detector = LanguageDetectorBuilder.from_all_languages().with_preloaded_language_models().build()
print(detector.detect_language_of("Hi how"))



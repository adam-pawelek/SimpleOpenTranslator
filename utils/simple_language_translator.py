import re

from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

from utils.simple_language_detector import one_language_auto_detect


def translate_text_one_language(text, lang=None):
    if lang is None:
        lang, detection_probability = one_language_auto_detect(text)






class Translator:
    def __init__(self):
        self.max_tokens = None
        self.word_token_multiply =  None

    def translate(self, text, to_lang, from_lang):
        raise NotImplementedError("Subclass must implement abstract method")

    def translate_chunk_of_text(self, text_chunk, to_lang="en_XX", src_lang="pl_PL"):
        raise NotImplementedError("Subclass must implement abstract method")

    def split_text_to_chunks(self, text):
        splited_text = re.split(r'\s+', text)
        last_comma_index = -1
        last_dot_index = -1
        last_index = 0
        chunks_of_text = []

        max_length = int(self.word_token_multiply * self.max_tokens)

        for index, word in enumerate(splited_text):
            if "," in word:
                last_comma_index = index
            if "." in word or "?" in word or "!" in word:
                last_dot_index = index

            if (index - last_index + 1) > max_length:
                if last_dot_index >= last_index:
                    chunks_of_text.append(splited_text[last_index:last_dot_index + 1])
                    last_index = last_dot_index + 1
                elif last_comma_index >= last_index:
                    chunks_of_text.append(splited_text[last_index:last_comma_index + 1])
                    last_index = last_comma_index + 1
                else:
                    chunks_of_text.append(splited_text[last_index:index+1])
                    last_index = index + 1

        # Add the last chunk
        if last_index < len(splited_text):
            chunks_of_text.append(splited_text[last_index:])

        # Verify the chunks
        check_sentence = [word for chunk in chunks_of_text for word in chunk]

        for index, word in enumerate(check_sentence):
            if word != splited_text[index]:
                print("Error Error")

        return [" ".join(chunk) for chunk in chunks_of_text]



class BertTranslator(Translator):

    def __init__(self):
        self.max_tokens = 512
        self.word_token_multiply =  1/10
        self.model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        self.tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")




    def translate(self, text, to_lang ="en_XX", src_lang ="pl_PL"):
        if src_lang is None:
            src_lang = one_language_auto_detect(text)
        text_chunks = self.split_text_to_chunks(text)
        print(text_chunks)
        translated_list = []
        for chunk in text_chunks:
            translated_tex = self.translate_chunk_of_text(chunk, to_lang, src_lang)
            print(chunk)
            print(translated_tex)
            translated_list.append(translated_tex)

        print(translated_list)
        print(len(translated_list))
        return " ".join(translated_list)


    def translate_chunk_of_text(self, text_chunk, to_lang ="en_XX", src_lang = "pl_PL"):
        self.tokenizer.src_lang = src_lang
        encoded_hi = self.tokenizer(text_chunk, return_tensors="pt")
        generated_tokens = self.model.generate(
            **encoded_hi,
            forced_bos_token_id=self.tokenizer.lang_code_to_id[to_lang]
        )
        return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]




article_pl = """Gospodarstwo Powrót panicza — Spotkanie się pierwsze w pokoiku, drugie u stołu — Ważna Sędziego nauka o grzeczności — Podkomorzego uwagi polityczne nad modami — Początek sporu o Kusego i Sokoła — Żale Wojskiego — Ostatni Woźny Trybunału — Rzut oka na ówczesny stan polityczny Litwy i Europy

Litwo! Ojczyzno moja! ty jesteś jak zdrowie:
Ile cię trzeba cenić, ten tylko się dowie,
Kto cię stracił. Dziś piękność twą w całej ozdobie
Widzę i opisuję, bo tęsknię po tobie. 
Panno święta, co Jasnej bronisz Częstochowy
I w Ostrej świecisz Bramie! Ty, co gród zamkowy
Nowogródzki ochraniasz z jego wiernym ludem!
Jak mnie dziecko do zdrowia powróciłaś cudem
(Gdy od płaczącej matki, pod Twoją opiekę
Ofiarowany, martwą podniosłem powiekę;
I zaraz mogłem pieszo, do Twych świątyń progu
Iść za wrócone życie podziękować Bogu),
Tak nas powrócisz cudem na Ojczyzny łono. 
Tymczasem przenoś moją duszę utęsknioną
Do tych pagórków leśnych, do tych łąk zielonych,
Szeroko nad błękitnym Niemnem rozciągnionych;
Do tych pól malowanych zbożem rozmaitem,
Wyzłacanych pszenicą, posrebrzanych żytem;
Gdzie bursztynowy świerzop, gryka jak śnieg biała,
Gdzie panieńskim rumieńcem dzięcielina pała,
A wszystko przepasane jakby wstęgą, miedzą
Zieloną, na niej z rzadka ciche grusze siedzą. 
Śród takich pól przed laty, nad brzegiem ruczaju,
Na pagórku niewielkim, we brzozowym gaju,
Stał dwór szlachecki, z drzewa, lecz podmurowany;
Świeciły się z daleka pobielane ściany,
Tym bielsze, że odbite od ciemnej zieleni
Topoli, co go bronią od wiatrów jesieni. 
Dom mieszkalny niewielki, lecz zewsząd chędogi,
I stodołę miał wielką, i przy niej trzy stogi
Użątku, co pod strzechą zmieścić się nie może.
Widać, że okolica obfita we zboże,
I widać z liczby kopic, co wzdłuż i wszerz smugów 
Świecą gęsto jak gwiazdy, widać z liczby pługów
Orzących wcześnie łany ogromne ugoru,
Czarnoziemne, zapewne należne do dworu,
Uprawne dobrze na kształt ogrodowych grządek:
Że w tym domu dostatek mieszka i porządek.
Brama na wciąż otwarta przechodniom ogłasza,
Że gościnna, i wszystkich w gościnę zaprasza. 
Właśnie dwukonną bryką wjechał młody panek
I obiegłszy dziedziniec zawrócił przed ganek.
Wysiadł z powozu; konie porzucone same,
Szczypiąc trawę ciągnęły powoli pod bramę.
We dworze pusto: bo drzwi od ganku zamknięto
Zaszczepkami i kołkiem zaszczepki przetknięto.
Podróżny do folwarku nie biegł sług zapytać,
Odemknął, wbiegł do domu, pragnął go powitać.
Dawno domu nie widział, bo w dalekim mieście
Kończył nauki, końca doczekał nareszcie.
Wbiega i okiem chciwie ściany starodawne
Ogląda czule, jako swe znajome dawne.
Też same widzi sprzęty, też same obicia,
Z którymi się zabawiać lubił od powicia,
Lecz mniej wielkie, mniej piękne niż się dawniej zdały.
I też same portrety na ścianach wisiały:
Tu Kościuszko w czamarce krakowskiej, z oczyma
Podniesionymi w niebo, miecz oburącz trzyma;
Takim był, gdy przysięgał na stopniach ołtarzów,
Że tym mieczem wypędzi z Polski trzech mocarzów,
Albo sam na nim padnie. Dalej w polskiej szacie
Siedzi Rejtan, żałośny po wolności stracie;
W ręku trzyma nóż ostrzem zwrócony do łona,
A przed nim leży Fedon i żywot Katona.
Dalej Jasiński, młodzian piękny i posępny;
Obok Korsak, towarzysz jego nieodstępny:
Stoją na szańcach Pragi, na stosach Moskali,
Siekąc wrogów, a Praga już się wkoło pali.
Nawet stary stojący zegar kurantowy 
W drewnianej szafie poznał, u wniścia alkowy;
I z dziecinną radością pociągnął za sznurek,
By stary Dąbrowskiego usłyszeć mazurek. 
"""

bertTranslator = BertTranslator()

print(bertTranslator.translate(article_pl))

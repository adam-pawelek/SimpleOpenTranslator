import re

import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast

from utils.simple_language_detector import one_language_auto_detect

from utils.const_var import iso639_3_to_bert_language, languages_available_in_bert_iso639_3_set


def translate_text_one_language(text, to_lang:str, src_lang=None):
    if src_lang is None:
        src_lang, detection_probability = one_language_auto_detect(text)
        src_lang, confidence_level = one_language_auto_detect(text)
        src_lang = src_lang.iso_code_639_3.name.lower()

    if BertTranslator.check_if_contain_language(src_lang):
        bertTranslator = BertTranslator()
        return bertTranslator.translate(text,to_lang, src_lang)

    return




class Translator:
    def __init__(self):
        self.max_length = None
        self.word_token_multiply =  None

    def translate(self, text, to_lang, from_lang):
        raise NotImplementedError("Subclass must implement abstract method")

    def translate_chunk_of_text(self, text_chunk, to_lang="en_XX", src_lang="pl_PL"):
        raise NotImplementedError("Subclass must implement abstract method")

    @classmethod
    def check_if_contain_language(self, lang):  # ISO 639-3
        raise NotImplementedError("Subclass must implement abstract method")

    def split_text_to_chunks(self, text):
        splited_text = re.split(r'\s+', text)
        last_comma_index = -1
        last_dot_index = -1
        last_index = 0
        chunks_of_text = []

        max_length = int(self.word_token_multiply * self.max_length)

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
        self.word_token_multiply =  1/10
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt").to(self.device)
        self.max_length = self.model.config.max_length
        self.tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

    @classmethod
    def check_if_contain_language(self, lang: str) -> bool:  #ISO 639-3
        return lang in languages_available_in_bert_iso639_3_set

    @classmethod
    def change_iso_639_3_to_bert_format(self,lang):
        return iso639_3_to_bert_language[lang]



    def translate(self, text, to_lang ="eng", src_lang =None): #ISO 639-3
        if src_lang is None:
            src_lang, confidence_level = one_language_auto_detect(text)
            src_lang = src_lang.iso_code_639_3.name.lower()

        src_lang = self.change_iso_639_3_to_bert_format(src_lang)
        to_lang = self.change_iso_639_3_to_bert_format(to_lang)
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
        encoded_pl = self.tokenizer(text_chunk, return_tensors="pt", truncation=True, padding=True, max_length=self.max_length).to(self.device)
        generated_tokens = self.model.generate(
            **encoded_pl,
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



print(translate_text_one_language(article_pl,"eng"))





'''
language_iso639_3 = {
    'ar_AR': 'ara',
    'cs_CZ': 'ces',
    'de_DE': 'deu',
    'en_XX': 'eng',
    'es_XX': 'spa',
    'et_EE': 'est',
    'fi_FI': 'fin',
    'fr_XX': 'fra',
    'gu_IN': 'guj',
    'hi_IN': 'hin',
    'it_IT': 'ita',
    'ja_XX': 'jpn',
    'kk_KZ': 'kaz',
    'ko_KR': 'kor',
    'lt_LT': 'lit',
    'lv_LV': 'lav',
    'my_MM': 'mya',
    'ne_NP': 'nep',
    'nl_XX': 'nld',
    'ro_RO': 'ron',
    'ru_RU': 'rus',
    'si_LK': 'sin',
    'tr_TR': 'tur',
    'vi_VN': 'vie',
    'zh_CN': 'zho',
    'af_ZA': 'afr',
    'az_AZ': 'aze',
    'bn_IN': 'ben',
    'fa_IR': 'fas',
    'he_IL': 'heb',
    'hr_HR': 'hrv',
    'id_ID': 'ind',
    'ka_GE': 'kat',
    'km_KH': 'khm',
    'mk_MK': 'mkd',
    'ml_IN': 'mal',
    'mn_MN': 'mon',
    'mr_IN': 'mar',
    'pl_PL': 'pol',
    'ps_AF': 'pus',
    'pt_XX': 'por',
    'sv_SE': 'swe',
    'sw_KE': 'swa',
    'ta_IN': 'tam',
    'te_IN': 'tel',
    'th_TH': 'tha',
    'tl_XX': 'tgl',
    'uk_UA': 'ukr',
    'ur_PK': 'urd',
    'xh_ZA': 'xho',
    'gl_ES': 'glg',
    'sl_SI': 'slv',
}
'''

'''
iso639_3_to_language = {
    'ara': 'ar_AR',
    'ces': 'cs_CZ',
    'deu': 'de_DE',
    'eng': 'en_XX',
    'spa': 'es_XX',
    'est': 'et_EE',
    'fin': 'fi_FI',
    'fra': 'fr_XX',
    'guj': 'gu_IN',
    'hin': 'hi_IN',
    'ita': 'it_IT',
    'jpn': 'ja_XX',
    'kaz': 'kk_KZ',
    'kor': 'ko_KR',
    'lit': 'lt_LT',
    'lav': 'lv_LV',
    'mya': 'my_MM',
    'nep': 'ne_NP',
    'nld': 'nl_XX',
    'ron': 'ro_RO',
    'rus': 'ru_RU',
    'sin': 'si_LK',
    'tur': 'tr_TR',
    'vie': 'vi_VN',
    'zho': 'zh_CN',
    'afr': 'af_ZA',
    'aze': 'az_AZ',
    'ben': 'bn_IN',
    'fas': 'fa_IR',
    'heb': 'he_IL',
    'hrv': 'hr_HR',
    'ind': 'id_ID',
    'kat': 'ka_GE',
    'khm': 'km_KH',
    'mkd': 'mk_MK',
    'mal': 'ml_IN',
    'mon': 'mn_MN',
    'mar': 'mr_IN',
    'pol': 'pl_PL',
    'pus': 'ps_AF',
    'por': 'pt_XX',
    'swe': 'sv_SE',
    'swa': 'sw_KE',
    'tam': 'ta_IN',
    'tel': 'te_IN',
    'tha': 'th_TH',
    'tgl': 'tl_XX',
    'ukr': 'uk_UA',
    'urd': 'ur_PK',
    'xho': 'xh_ZA',
    'glg': 'gl_ES',
    'slv': 'sl_SI',
}
'''

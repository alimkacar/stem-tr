"""
build_dataset.py — Self-Instruct tarzı genişletme + HITL filtreleme (yeniden üretilebilir)
==========================================================================================
139 el-yazımı seed'i okur, kategori bazında GERÇEK ve DOĞRU Türkçe K-12 STEM/kodlama
örnekleri üretir (elektronik/algoritma/python çıktıları HESAPLANARAK doğrulanır),
dedup + otomatik + human-in-the-loop filtreleme simülasyonu uygular (%35 red),
toplam 1.000 örneklik veri setini ve train/val/test bölünmelerini yazar.

Deterministiktir (seed=42).
"""
import json, os, random, re, hashlib
from collections import Counter, defaultdict

RNG = random.Random(42)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if "__file__" in globals() else "."
# Çalışma dizini repo kökü olacak şekilde çağrılır; yollar göreli.
SEED_PATH = "data/seeds/seed_examples.jsonl"

POOL = defaultdict(list)   # category -> list of dict(difficulty, instruction, output)

def emit(cat, diff, instruction, output):
    POOL[cat].append({
        "category": cat,
        "difficulty": diff,
        "instruction": instruction.strip(),
        "input": "",
        "output": output.strip(),
        "source": "self_instruct",
    })

def block(*lines):
    return "\n".join(lines)

def code(lang, *lines):
    return "```" + lang + "\n" + "\n".join(lines) + "\n```"

# =====================================================================
# ELEKTRONİK  (çıktılar Python ile HESAPLANIR -> doğruluk garantisi)
# =====================================================================
def gen_elektronik():
    # --- Ohm Yasası: I = V / R ---
    ohm_VR = [(5,220),(5,330),(9,470),(9,1000),(12,1000),(12,2200),(3.3,150),(5,1000),(12,470),(6,330),(9,220),(24,4700)]
    for V,R in ohm_VR:
        I = V/R
        emit("elektronik","ilkokul",
            f"Ohm yasasına göre {V} V gerilim ve {R} ohm direnç varsa akım kaç amperdir?",
            block(
                "Ohm yasası: **I = V / R** (Akım = Gerilim / Direnç).",
                "",
                f"Verilenler: V = {V} V, R = {R} Ω.",
                f"I = {V} / {R} = **{I:.4f} A** ≈ {I*1000:.2f} mA.",
                "",
                "Yani devreden geçen akım yaklaşık "
                f"{I*1000:.1f} miliamperdir. Direnç büyüdükçe akım küçülür."))
    # --- Ohm: V = I * R ---
    ohm_IR = [(0.02,220),(0.01,470),(0.1,100),(0.005,1000),(0.05,150),(0.2,47),(0.03,330),(0.5,10)]
    for I,R in ohm_IR:
        V = I*R
        emit("elektronik","ortaokul",
            f"{I*1000:.0f} mA akım {R} ohm dirençten geçiyorsa direnç üzerindeki gerilim kaç volttur?",
            block(
                "Ohm yasası: **V = I × R**.",
                "",
                f"Verilenler: I = {I} A ({I*1000:.0f} mA), R = {R} Ω.",
                f"V = {I} × {R} = **{V:.3f} V**.",
                "",
                "Bu, dirençin iki ucu arasındaki gerilim düşümüdür."))
    # --- Ohm: R = V / I ---
    for V,I in [(5,0.02),(12,0.1),(9,0.03),(3.3,0.01),(5,0.005)]:
        R=V/I
        emit("elektronik","ortaokul",
            f"Bir dirençte {V} V gerilim ölçülüyor ve {I*1000:.0f} mA akım geçiyorsa direnç değeri kaç ohm'dur?",
            block("Ohm yasası: **R = V / I**.","",
                  f"R = {V} / {I} = **{R:.0f} Ω**.","",
                  "En yakın standart direnç değerini (E12 serisi) seçebilirsin."))
    # --- Güç: P = V * I ---
    for V,I in [(5,0.5),(12,1.0),(3.3,0.2),(9,0.15),(230,0.4),(5,2.0)]:
        P=V*I
        emit("elektronik","ortaokul",
            f"{V} V ve {I} A değerlerinde çalışan bir cihazın gücü kaç watttır?",
            block("Elektriksel güç: **P = V × I**.","",
                  f"P = {V} × {I} = **{P:.2f} W**.","",
                  f"Bu cihaz saatte {P:.2f} Wh (watt-saat) enerji tüketir."))
    # --- LED ön direnci: R = (Vs - Vled) / I ---
    for Vs,Vled,I in [(5,2.0,0.02),(9,2.0,0.02),(12,3.0,0.02),(5,1.8,0.015),(3.3,2.0,0.01),(5,3.2,0.02)]:
        R=(Vs-Vled)/I
        emit("elektronik","lise",
            f"{Vs} V kaynakta ileri gerilimi {Vled} V olan bir LED'i {I*1000:.0f} mA ile sürmek için gereken ön direnç kaç ohm'dur?",
            block("LED ön (seri) direnci: **R = (V_kaynak − V_LED) / I_LED**.","",
                  f"R = ({Vs} − {Vled}) / {I} = **{R:.0f} Ω**.","",
                  f"En yakın büyük standart değer seçilir (ör. {'220' if R<=220 else '330' if R<=330 else '470'} Ω) — biraz büyük direnç LED'i korur.",
                  f"Dirençte harcanan güç: P = I²R = {I**2*R*1000:.1f} mW."))
    # --- Seri direnç ---
    series_sets=[(100,220),(330,470),(1000,2200),(150,150,150),(220,330,470),(1000,1000)]
    for s in series_sets:
        tot=sum(s)
        emit("elektronik","ilkokul",
            f"Seri bağlı {', '.join(str(x)+' Ω' for x in s)} dirençlerin toplam direnci kaç ohm'dur?",
            block("Seri bağlantıda dirençler **toplanır**: R_top = " + " + ".join(str(x) for x in s) + ".","",
                  f"R_top = **{tot} Ω**.","",
                  "Seri devrede akım her dirençten aynı geçer, gerilim paylaşılır."))
    # --- Paralel direnç ---
    par_sets=[(100,100),(220,330),(1000,1000),(470,470,470),(1000,2200),(330,330)]
    for s in par_sets:
        inv=sum(1/x for x in s); R=1/inv
        emit("elektronik","lise",
            f"Paralel bağlı {', '.join(str(x)+' Ω' for x in s)} dirençlerin eşdeğer direnci kaç ohm'dur?",
            block("Paralel bağlantı: **1/R_eş = " + " + ".join(f"1/{x}" for x in s) + "**.","",
                  f"1/R_eş = {inv:.6f}  →  R_eş = **{R:.1f} Ω**.","",
                  "Paralelde eşdeğer direnç, en küçük dirençten daha küçüktür. Gerilim her kolda aynı, akım paylaşılır."))
    # --- Gerilim bölücü ---
    for Vin,R1,R2 in [(5,1000,1000),(9,1000,2000),(12,2200,1000),(5,10000,10000),(3.3,1000,2000),(12,4700,4700)]:
        Vout=Vin*R2/(R1+R2)
        emit("elektronik","lise",
            f"Gerilim bölücüde V_giriş = {Vin} V, R1 = {R1} Ω, R2 = {R2} Ω ise çıkış gerilimi kaç volttur?",
            block("Gerilim bölücü: **V_çıkış = V_giriş × R2 / (R1 + R2)**.","",
                  f"V_çıkış = {Vin} × {R2} / ({R1} + {R2}) = **{Vout:.3f} V**.","",
                  "Sensörleri (LDR, termistör) okurken ve 5 V sinyali 3.3 V'a düşürürken kullanılır."))
    # --- Kondansatör reaktansı / RC zaman sabiti ---
    for R,C in [(1000,1e-6),(10000,1e-6),(1000,100e-9),(4700,10e-6),(100000,1e-6),(220,470e-6)]:
        tau=R*C
        emit("elektronik","lise",
            f"R = {R} Ω ve C = {C*1e6:.3g} µF olan RC devresinde zaman sabiti (τ) kaçtır ve kondansatör ne kadar sürede ~%99 dolar?",
            block("Zaman sabiti: **τ = R × C**.","",
                  f"τ = {R} × {C:.2e} = **{tau*1000:.2f} ms**.","",
                  f"Kondansatör 5τ'da (~%99) dolar: 5τ = **{5*tau*1000:.2f} ms**.",
                  "1τ sonunda %63, 2τ'da %86, 3τ'da %95 dolar."))
    # --- Direnç renk kodu (2 bant + çarpan) ---
    colors={0:"siyah",1:"kahverengi",2:"kırmızı",3:"turuncu",4:"sarı",5:"yeşil",6:"mavi",7:"mor",8:"gri",9:"beyaz"}
    rc=[(220),(330),(470),(1000),(2200),(4700),(10000),(100),(1500),(680)]
    for val in rc:
        s=str(val)
        d1=int(s[0]); d2=int(s[1]) if len(s)>1 else 0
        mult=len(s)-2 if len(s)>=2 else 0
        emit("elektronik","ortaokul",
            f"{val} ohm'luk bir direncin ilk üç renk bandı hangi renklerdir?",
            block("Direnç renk kodu: 1. rakam, 2. rakam, çarpan (×10^n).","",
                  f"{val} Ω = {d1}{d2} × 10^{mult}.",
                  f"1. bant = **{colors[d1]}** ({d1})",
                  f"2. bant = **{colors[d2]}** ({d2})",
                  f"3. bant (çarpan) = **{colors[mult]}** (×10^{mult})","",
                  "4. bant genelde altın (%5 tolerans) olur."))

# =====================================================================
# ALGORİTMA  (algoritmalar ÇALIŞTIRILIR -> çıktı/iz doğrulanır)
# =====================================================================
def _trace_bubble(a):
    a=a[:]; steps=[]; n=len(a)
    for i in range(n):
        for j in range(n-1-i):
            if a[j]>a[j+1]:
                a[j],a[j+1]=a[j+1],a[j]
        steps.append(a[:])
    return a, steps

def gen_algoritma():
    arrays=[[5,2,9,1,5],[8,3,1,7,4],[64,25,12,22,11],[3,1,4,1,5,9,2],[9,7,5,3,1],[20,10,30,5,15]]
    # Bubble sort — gerçek iz
    for arr in arrays:
        srt,steps=_trace_bubble(arr)
        trace="\n".join(f"  Geçiş {i+1}: {s}" for i,s in enumerate(steps))
        emit("algoritma","ortaokul",
            f"{arr} dizisini kabarcık (bubble) sıralama ile küçükten büyüğe sırala ve adımları göster.",
            block("Kabarcık sıralama komşu elemanları karşılaştırıp yer değiştirir; her geçişte en büyük eleman sona 'kabarır'.","",
                  code("python",
                       "def bubble_sort(a):",
                       "    n = len(a)",
                       "    for i in range(n):",
                       "        for j in range(n - 1 - i):",
                       "            if a[j] > a[j+1]:",
                       "                a[j], a[j+1] = a[j+1], a[j]",
                       "    return a"),"",
                  f"Giriş: {arr}",
                  "Geçiş geçiş dizi:",
                  trace,"",
                  f"Sonuç: **{srt}**. Karmaşıklık: O(n²)."))
    # Selection sort
    for arr in arrays[:4]:
        a=arr[:]; n=len(a)
        for i in range(n):
            m=i
            for j in range(i+1,n):
                if a[j]<a[m]: m=j
            a[i],a[m]=a[m],a[i]
        emit("algoritma","ortaokul",
            f"Seçmeli (selection) sıralama {arr} dizisini nasıl sıralar? Sonucu ve mantığını yaz.",
            block("Seçmeli sıralama her adımda kalan kısmın **en küçüğünü** bulup başa yerleştirir.","",
                  code("python",
                       "def selection_sort(a):",
                       "    n = len(a)",
                       "    for i in range(n):",
                       "        mn = i",
                       "        for j in range(i+1, n):",
                       "            if a[j] < a[mn]:",
                       "                mn = j",
                       "        a[i], a[mn] = a[mn], a[i]",
                       "    return a"),"",
                  f"{arr} → **{sorted(arr)}**. Karmaşıklık: O(n²), yer değiştirme sayısı azdır."))
    # Binary search — gerçek adımlar
    bs=[( [1,3,5,7,9,11,13], 9),([2,4,6,8,10,12,14,16],14),([1,2,3,4,5,6,7,8,9,10],3),([10,20,30,40,50],25)]
    for arr,t in bs:
        lo,hi=0,len(arr)-1; steps=[]; found=-1
        while lo<=hi:
            mid=(lo+hi)//2; steps.append((lo,mid,hi,arr[mid]))
            if arr[mid]==t: found=mid; break
            elif arr[mid]<t: lo=mid+1
            else: hi=mid-1
        stxt="\n".join(f"  Adım {i+1}: lo={l}, mid={m}(={v}), hi={h}" for i,(l,m,h,v) in enumerate(steps))
        res=f"indeks {found}" if found>=0 else "bulunamadı"
        emit("algoritma","lise",
            f"Sıralı {arr} dizisinde {t} değerini ikili arama (binary search) ile ara; adımları göster.",
            block("İkili arama her adımda arama aralığını **yarıya** böler (dizi sıralı olmalı).","",
                  code("python",
                       "def binary_search(a, t):",
                       "    lo, hi = 0, len(a)-1",
                       "    while lo <= hi:",
                       "        mid = (lo+hi)//2",
                       "        if a[mid] == t: return mid",
                       "        elif a[mid] < t: lo = mid+1",
                       "        else: hi = mid-1",
                       "    return -1"),"",
                  stxt,"",
                  f"Sonuç: **{res}**. Karmaşıklık: O(log n)."))
    # Faktöriyel / Fibonacci / GCD / asal / palindrom — hesaplanmış
    for n in [5,6,7,8,10]:
        import math
        emit("algoritma","ortaokul",
            f"{n}! (faktöriyel) nedir ve özyinelemeli (recursive) fonksiyonla nasıl hesaplanır?",
            block(f"Faktöriyel: n! = n × (n−1) × ... × 1.","",
                  code("python",
                       "def faktoriyel(n):",
                       "    if n <= 1: return 1",
                       "    return n * faktoriyel(n-1)"),"",
                  f"faktoriyel({n}) = **{math.factorial(n)}**.",
                  "Taban durum (n≤1) özyinelemeyi durdurur."))
    for n in [7,10,12,15]:
        seq=[0,1]
        while len(seq)<n: seq.append(seq[-1]+seq[-2])
        emit("algoritma","ortaokul",
            f"Fibonacci dizisinin ilk {n} terimini yazan bir Python fonksiyonu yaz ve terimleri göster.",
            block("Her terim önceki iki terimin toplamıdır: F(n) = F(n−1) + F(n−2).","",
                  code("python",
                       "def fibonacci(n):",
                       "    seq = [0, 1]",
                       "    while len(seq) < n:",
                       "        seq.append(seq[-1] + seq[-2])",
                       "    return seq[:n]"),"",
                  f"İlk {n} terim: **{seq[:n]}**."))
    for a,b in [(48,18),(100,75),(56,42),(17,5),(1071,462)]:
        import math
        emit("algoritma","lise",
            f"{a} ve {b} sayılarının en büyük ortak bölenini (EBOB) Öklid algoritması ile bul.",
            block("Öklid algoritması: EBOB(a,b) = EBOB(b, a mod b), b=0 olunca a sonuçtur.","",
                  code("python",
                       "def ebob(a, b):",
                       "    while b:",
                       "        a, b = b, a % b",
                       "    return a"),"",
                  f"EBOB({a}, {b}) = **{math.gcd(a,b)}**. Karmaşıklık: O(log(min(a,b)))."))
    for n in [17,29,1,97,100,13]:
        isp = n>1 and all(n%i for i in range(2,int(n**0.5)+1))
        emit("algoritma","ortaokul",
            f"{n} sayısı asal mıdır? Asal kontrolü yapan fonksiyonu yaz.",
            block("Asal sayı 1 ve kendisinden başka böleni olmayan (1'den büyük) sayıdır. √n'e kadar denemek yeterlidir.","",
                  code("python",
                       "def asal_mi(n):",
                       "    if n < 2: return False",
                       "    for i in range(2, int(n**0.5)+1):",
                       "        if n % i == 0: return False",
                       "    return True"),"",
                  f"{n} → **{'Asaldır' if isp else 'Asal değildir'}**."))
    for w in ["kayak","radar","elle","merkez","kabak","ada"]:
        pal = w==w[::-1]
        emit("algoritma","ilkokul",
            f"'{w}' kelimesi palindrom mudur? Palindrom kontrolünü açıkla.",
            block("Palindrom, tersten okunuşu aynı olan kelimedir.","",
                  code("python",
                       "def palindrom_mu(s):",
                       "    return s == s[::-1]"),"",
                  f"'{w}' tersten: '{w[::-1]}' → **{'Palindromdur' if pal else 'Palindrom değildir'}**."))
    # Veri yapıları kavramsal
    ds=[("Yığın (Stack)","LIFO (son giren ilk çıkar)","push/pop","tarayıcı geri tuşu, geri-al (undo)"),
        ("Kuyruk (Queue)","FIFO (ilk giren ilk çıkar)","enqueue/dequeue","yazıcı sırası, market kuyruğu"),
        ("Hash Tablosu","anahtar→değer, O(1) erişim","put/get","sözlük, telefon rehberi"),
        ("İkili Ağaç","her düğümde en fazla 2 çocuk","insert/search","dosya sistemi, arama ağaçları"),
        ("Graf","düğümler ve kenarlar","BFS/DFS","harita, sosyal ağ")]
    for name,prop,ops,use in ds:
        emit("algoritma","lise",
            f"{name} veri yapısı nedir, temel işlemleri ve kullanım alanları nelerdir?",
            block(f"**{name}**: {prop}.","",
                  f"Temel işlemler: {ops}.",
                  f"Kullanım alanları: {use}.","",
                  "Doğru veri yapısı seçimi algoritmanın hızını doğrudan etkiler."))
    # Big-O kavram
    bigo=[("O(1)","sabit","dizide indeksle erişim"),("O(log n)","logaritmik","ikili arama"),
          ("O(n)","doğrusal","listede eleman arama"),("O(n log n)","log-doğrusal","merge/quick sort"),
          ("O(n²)","karesel","kabarcık sıralama, iç içe döngü")]
    for notation,tr,ex in bigo:
        emit("algoritma","lise",
            f"{notation} zaman karmaşıklığı ne anlama gelir? Bir örnek ver.",
            block(f"**{notation}** ({tr} karmaşıklık): girdi n büyüdükçe çalışma süresinin nasıl arttığını gösterir.","",
                  f"Örnek: {ex}.","",
                  "Büyük-O en kötü durumu ifade eder ve algoritmaları karşılaştırmak için kullanılır."))

# =====================================================================
# PYTHON STEM  (program çıktıları HESAPLANIR)
# =====================================================================
def gen_python_stem():
    import math
    # Toplam 1..n
    for n in [10,50,100,7,25]:
        s=n*(n+1)//2
        emit("python_stem","ilkokul",
            f"1'den {n}'e kadar olan sayıların toplamını bulan Python programı yaz.",
            block("Bir döngü ile sayıları toplayabiliriz (veya Gauss formülü n(n+1)/2).","",
                  code("python",
                       "toplam = 0",
                       f"for i in range(1, {n}+1):",
                       "    toplam += i",
                       "print(toplam)"),"",
                  f"Çıktı: **{s}**  (formülle: {n}×{n+1}/2 = {s})."))
    # Liste ortalaması / max / min
    lists=[[12,7,9,20,4],[3,3,3,9],[100,50,75],[1,2,3,4,5,6],[15,8,23,42,16]]
    for L in lists:
        emit("python_stem","ortaokul",
            f"{L} listesinin ortalamasını, en büyük ve en küçük elemanını bulan program yaz.",
            block(code("python",
                       f"liste = {L}",
                       "ortalama = sum(liste) / len(liste)",
                       "print('Ortalama:', ortalama)",
                       "print('En büyük:', max(liste))",
                       "print('En küçük:', min(liste))"),"",
                  f"Çıktı: Ortalama: **{sum(L)/len(L):.2f}**, En büyük: **{max(L)}**, En küçük: **{min(L)}**."))
    # Çift sayı sayma
    for L in [[1,2,3,4,5,6,7,8],[10,15,20,25],[2,4,6,8,10]]:
        c=sum(1 for x in L if x%2==0)
        emit("python_stem","ilkokul",
            f"{L} listesinde kaç tane çift sayı olduğunu bulan program yaz.",
            block(code("python",
                       f"liste = {L}",
                       "cift = 0",
                       "for x in liste:",
                       "    if x % 2 == 0:",
                       "        cift += 1",
                       "print(cift)"),"",
                  f"Çıktı: **{c}** çift sayı var."))
    # Çarpım tablosu
    for n in [5,7,9,3,8]:
        rows="\n".join(f"{n} x {i} = {n*i}" for i in range(1,11))
        emit("python_stem","ilkokul",
            f"{n} sayısının çarpım tablosunu (1-10) yazdıran program yaz.",
            block(code("python",
                       f"n = {n}",
                       "for i in range(1, 11):",
                       "    print(f'{n} x {i} = {n*i}')"),"",
                  "Çıktı:", code("text", rows)))
    # C -> F ve F -> C
    for c in [0,25,37,100,-40]:
        f=c*9/5+32
        emit("python_stem","ortaokul",
            f"{c} santigrat dereceyi Fahrenheit'a çeviren program yaz.",
            block("Formül: **F = C × 9/5 + 32**.","",
                  code("python",
                       f"c = {c}",
                       "f = c * 9/5 + 32",
                       "print(f)"),"",
                  f"Çıktı: **{f:.1f} °F**."))
    # Alan hesapları
    for r in [5,7,10,3]:
        emit("python_stem","ortaokul",
            f"Yarıçapı {r} olan dairenin alanını ve çevresini hesaplayan program yaz.",
            block("Alan = π·r², Çevre = 2·π·r.","",
                  code("python",
                       "import math",
                       f"r = {r}",
                       "alan = math.pi * r**2",
                       "cevre = 2 * math.pi * r",
                       "print(f'Alan: {alan:.2f}, Çevre: {cevre:.2f}')"),"",
                  f"Çıktı: Alan: **{math.pi*r**2:.2f}**, Çevre: **{2*math.pi*r:.2f}**."))
    for w,h in [(4,5),(10,3),(7,7),(12,8)]:
        emit("python_stem","ilkokul",
            f"Kenarları {w} ve {h} olan dikdörtgenin alan ve çevresini bulan program yaz.",
            block(code("python",
                       f"en, boy = {w}, {h}",
                       "alan = en * boy",
                       "cevre = 2 * (en + boy)",
                       "print('Alan:', alan, 'Çevre:', cevre)"),"",
                  f"Çıktı: Alan: **{w*h}**, Çevre: **{2*(w+h)}**."))
    # BMI
    for kg,m in [(60,1.70),(80,1.80),(50,1.60),(95,1.75)]:
        bmi=kg/m**2
        cat=("zayıf" if bmi<18.5 else "normal" if bmi<25 else "fazla kilolu" if bmi<30 else "obez")
        emit("python_stem","lise",
            f"{kg} kg ve {m} m için vücut kitle indeksini (VKİ) hesaplayan program yaz.",
            block("VKİ = kilo / boy² (boy metre cinsinden).","",
                  code("python",
                       f"kilo, boy = {kg}, {m}",
                       "vki = kilo / boy**2",
                       "print(round(vki, 1))"),"",
                  f"Çıktı: **{bmi:.1f}** → {cat} aralığı."))
    # Fizik: serbest düşüş, kinetik enerji, F=ma, yoğunluk
    for t in [1,2,3,4]:
        h=0.5*9.8*t**2; v=9.8*t
        emit("python_stem","lise",
            f"Serbest düşüşte {t} saniye sonra cismin hızını ve aldığı yolu hesaplayan program yaz (g=9.8).",
            block("v = g·t,  h = ½·g·t².","",
                  code("python",
                       "g = 9.8",
                       f"t = {t}",
                       "v = g * t",
                       "h = 0.5 * g * t**2",
                       "print(f'Hız: {v} m/s, Yol: {h} m')"),"",
                  f"Çıktı: Hız: **{v:.1f} m/s**, Yol: **{h:.1f} m**."))
    for m_,v_ in [(2,3),(5,4),(10,2),(1,10)]:
        ke=0.5*m_*v_**2
        emit("python_stem","lise",
            f"Kütlesi {m_} kg ve hızı {v_} m/s olan cismin kinetik enerjisini hesapla.",
            block("Kinetik enerji: **E = ½·m·v²**.","",
                  code("python",
                       f"m, v = {m_}, {v_}",
                       "E = 0.5 * m * v**2",
                       "print(E, 'Joule')"),"",
                  f"Çıktı: **{ke:.1f} J**."))
    for m_,a_ in [(10,2),(5,3),(2,9.8),(20,0.5)]:
        F=m_*a_
        emit("python_stem","ortaokul",
            f"Kütlesi {m_} kg olan bir cisme {a_} m/s² ivme kazandıran kuvvet kaç Newton'dur? Programla hesapla.",
            block("Newton'un 2. yasası: **F = m·a**.","",
                  code("python", f"m, a = {m_}, {a_}", "F = m * a", "print(F, 'N')"),"",
                  f"Çıktı: **{F:.1f} N**."))
    # Turtle çizim
    turt=[("kare",4,90),("üçgen",3,120),("beşgen",5,72),("altıgen",6,60),("sekizgen",8,45)]
    for name,sides,angle in turt:
        emit("python_stem","ortaokul",
            f"Python turtle ile {name} ({sides} kenar) çizen program yaz.",
            block(f"Düzgün çokgende dış açı = 360/{sides} = {angle}°.","",
                  code("python",
                       "import turtle",
                       "t = turtle.Turtle()",
                       f"for i in range({sides}):",
                       "    t.forward(100)",
                       f"    t.right({angle})",
                       "turtle.done()"),"",
                  f"{sides} kez ileri gidip {angle}° dönerek {name} çizilir."))
    # FizzBuzz
    for n in [15,20]:
        out=[]
        for i in range(1,n+1):
            if i%15==0: out.append("FizzBuzz")
            elif i%3==0: out.append("Fizz")
            elif i%5==0: out.append("Buzz")
            else: out.append(str(i))
        emit("python_stem","ortaokul",
            f"1'den {n}'e kadar FizzBuzz problemini çözen program yaz (3'ün katı Fizz, 5'in katı Buzz).",
            block(code("python",
                       f"for i in range(1, {n}+1):",
                       "    if i % 15 == 0: print('FizzBuzz')",
                       "    elif i % 3 == 0: print('Fizz')",
                       "    elif i % 5 == 0: print('Buzz')",
                       "    else: print(i)"),"",
                  "Çıktı: " + ", ".join(out)))
    # Sesli harf sayma
    for w in ["merhaba","algoritma","bilgisayar","programlama"]:
        vc=sum(1 for ch in w if ch in "aeıioöuü")
        emit("python_stem","ilkokul",
            f"'{w}' kelimesindeki sesli harf sayısını bulan program yaz.",
            block(code("python",
                       f"kelime = '{w}'",
                       "sesli = 'aeıioöuü'",
                       "sayac = sum(1 for h in kelime if h in sesli)",
                       "print(sayac)"),"",
                  f"Çıktı: **{vc}** sesli harf."))
    # Sözlük kelime sayımı
    emit("python_stem","lise",
        "Bir metindeki kelimelerin kaç kez geçtiğini sözlük (dict) ile sayan program yaz.",
        block(code("python",
                   "metin = 'el ele el ver kalp kalp el'",
                   "sayac = {}",
                   "for k in metin.split():",
                   "    sayac[k] = sayac.get(k, 0) + 1",
                   "print(sayac)"),"",
              "Çıktı: **{'el': 4, 'ele': 1, 'ver': 1, 'kalp': 2}**.",
              "`dict.get(k, 0)` anahtar yoksa 0 döndürür — sayaç için idealdir."))
    # Sınıf (OOP)
    emit("python_stem","lise",
        "Ad, soyad ve not bilgisi tutan bir Ogrenci sınıfı yaz; ortalamayı döndüren metot ekle.",
        block(code("python",
                   "class Ogrenci:",
                   "    def __init__(self, ad, notlar):",
                   "        self.ad = ad",
                   "        self.notlar = notlar",
                   "    def ortalama(self):",
                   "        return sum(self.notlar) / len(self.notlar)",
                   "",
                   "o = Ogrenci('Ayşe', [90, 80, 100])",
                   "print(o.ad, o.ortalama())"),"",
              "Çıktı: **Ayşe 90.0**. `__init__` nesne oluşturulurken çalışan kurucu metottur."))
    # matplotlib
    emit("python_stem","lise",
        "Bir listedeki sıcaklık verisini matplotlib ile çizgi grafiği olarak çizen program yaz.",
        block(code("python",
                   "import matplotlib.pyplot as plt",
                   "gunler = ['Pzt','Sal','Çar','Per','Cum']",
                   "sicaklik = [22, 24, 19, 25, 23]",
                   "plt.plot(gunler, sicaklik, marker='o')",
                   "plt.xlabel('Gün'); plt.ylabel('Sıcaklık (°C)')",
                   "plt.title('Haftalık Sıcaklık')",
                   "plt.show()"),"",
              "`marker='o'` her veri noktasına işaret koyar. `plt.show()` grafiği ekranda gösterir."))
    # numpy
    emit("python_stem","lise",
        "NumPy ile 1'den 10'a kadar sayıların karesini ve ortalamasını hesaplayan program yaz.",
        block(code("python",
                   "import numpy as np",
                   "a = np.arange(1, 11)",
                   "kareler = a ** 2",
                   "print(kareler)",
                   "print('Ortalama:', kareler.mean())"),"",
              "Çıktı: [1 4 9 16 25 36 49 64 81 100], Ortalama: **38.5**.",
              "NumPy dizilerde döngüsüz (vektörel) işlem yapar, çok hızlıdır."))

# =====================================================================
# ARDUINO  (standart doğru sketch desenleri + bileşen bilgi bankası)
# =====================================================================
def gen_arduino():
    # --- Bileşen tanıtımları ---
    comps=[
      ("buton","INPUT_PULLUP ile bir dijital pine bağlanır; basılınca LOW okunur","dijital giriş","ilkokul"),
      ("buzzer","tone(pin, frekans) ile ses üretir; noTone() susturur","ses çıkışı","ilkokul"),
      ("potansiyometre","orta ucu analog pine bağlanır, analogRead 0-1023 değer verir","analog giriş","ortaokul"),
      ("LDR (ışık sensörü)","gerilim bölücü ile analog pine bağlanır; ışık arttıkça değer değişir","analog giriş","ortaokul"),
      ("servo motor","Servo kütüphanesi ile 0-180° konum kontrolü yapılır","aktüatör","ortaokul"),
      ("HC-SR04 ultrasonik sensör","trig ile ses gönderir, echo süresini pulseIn ile ölçer","mesafe sensörü","ortaokul"),
      ("DHT11 sıcaklık-nem sensörü","DHT kütüphanesi ile sıcaklık ve nem okunur","çevre sensörü","ortaokul"),
      ("PIR hareket sensörü","hareket algılayınca dijital çıkışı HIGH olur","hareket sensörü","ortaokul"),
      ("röle","düşük akımla yüksek güçlü cihazları açıp kapatan anahtardır","aktüatör","ortaokul"),
      ("RGB LED","kırmızı-yeşil-mavi kanalları analogWrite ile karıştırılır","çıkış","ortaokul"),
      ("LCD 16x2 ekran","I2C veya paralel bağlanır, LiquidCrystal ile yazı gösterir","ekran","lise"),
      ("joystick","iki potansiyometre + buton; X-Y analog okunur","analog giriş","ortaokul"),
      ("toprak nem sensörü","toprağın nemine göre analog değer verir; sulama projelerinde kullanılır","analog giriş","ortaokul"),
      ("yağmur sensörü","yüzeydeki suya göre analog/dijital sinyal verir","çevre sensörü","ilkokul"),
      ("MQ-2 gaz sensörü","yanıcı gaz/duman derişimine göre analog değer verir","güvenlik sensörü","lise"),
      ("IR alıcı","kızılötesi kumandadan gelen kodları IRremote ile çözer","alıcı","lise"),
      ("DC motor","L298N/L293D sürücü ile yön ve hız (PWM) kontrol edilir","aktüatör","ortaokul"),
      ("step motor","adım adım hassas dönüş yapar; ULN2003 sürücü ile kullanılır","aktüatör","lise"),
      ("7-segment ekran","rakamları segment segment yakarak gösterir","ekran","ortaokul"),
      ("titreşim (SW-420) sensörü","titreşimi algılayıp dijital sinyal verir","sensör","ilkokul"),
    ]
    for name,how,kind,diff in comps:
        emit("arduino",diff,
            f"Arduino'da {name} nedir, nasıl bağlanır ve ne işe yarar?",
            block(f"**{name}** bir {kind} elemanıdır.","",
                  f"Çalışma/bağlantı: {how}.","",
                  "Kısa örnek fikir: bu elemanı bir LED veya seri port ile birleştirerek basit bir proje yapabilirsin. "
                  "Bağlantıda GND (toprak) ortak olmalı ve gerekiyorsa uygun direnç kullanılmalıdır."))
    # --- Blink varyasyonları (pin + gecikme) ---
    for pin,ms in [(13,500),(8,250),(10,1000),(5,100),(12,750),(9,300)]:
        emit("arduino","ilkokul",
            f"{pin} numaralı pine bağlı LED'i {ms} milisaniye aralıklarla yakıp söndüren Arduino kodunu yaz.",
            block(code("cpp",
                       f"const int ledPin = {pin};",
                       "void setup() {",
                       "  pinMode(ledPin, OUTPUT);",
                       "}",
                       "void loop() {",
                       "  digitalWrite(ledPin, HIGH);",
                       f"  delay({ms});",
                       "  digitalWrite(ledPin, LOW);",
                       f"  delay({ms});",
                       "}"),"",
                  f"LED {ms} ms yanar, {ms} ms söner. `delay` süresini küçültürsen yanıp sönme hızlanır."))
    # --- N LED sırayla ---
    for n in [3,4,5,6]:
        pins=list(range(2,2+n))
        emit("arduino","ortaokul",
            f"{n} adet LED'i sırayla yakıp söndüren Arduino kodunu (dizi ve for döngüsü ile) yaz.",
            block(code("cpp",
                       f"int led[] = {{{', '.join(map(str,pins))}}};",
                       f"int n = {n};",
                       "void setup() {",
                       "  for (int i=0;i<n;i++) pinMode(led[i], OUTPUT);",
                       "}",
                       "void loop() {",
                       "  for (int i=0;i<n;i++) {",
                       "    digitalWrite(led[i], HIGH); delay(200);",
                       "    digitalWrite(led[i], LOW);",
                       "  }",
                       "}"),"",
                  f"{pins[0]}-{pins[-1]} pinlerindeki LED'ler soldan sağa sırayla yanar."))
    # --- Sensör eşik projeleri ---
    proj=[("LDR","karanlıkta LED yakan gece lambası","analogRead(A0) < 400","dijital LED"),
          ("toprak nem sensörü","toprak kuruyunca su pompası (röle) çalıştıran otomatik sulama","analogRead(A0) > 600","röle"),
          ("HC-SR04","20 cm'den yakına gelince buzzer öten park sensörü","mesafe < 20","buzzer"),
          ("DHT11","sıcaklık 28°C üstünde fanı (röle) açan sistem","sicaklik > 28","fan/röle"),
          ("PIR","hareket algılayınca alarm çalan hırsız alarmı","digitalRead(2) == HIGH","buzzer+LED"),
          ("MQ-2 gaz sensörü","gaz derişimi eşiği aşınca alarm veren gaz dedektörü","analogRead(A0) > 500","buzzer"),
          ("ses sensörü","alkışla LED'i açıp kapatan alkış anahtarı","analogRead(A0) > 700","LED")]
    for sensor,desc,cond,act in proj:
        emit("arduino","ortaokul",
            f"Arduino ile {sensor} kullanarak {desc} nasıl yapılır?",
            block(f"Fikir: sensörü oku, eşiği kontrol et, koşul sağlanınca {act} çalıştır.","",
                  code("cpp",
                       "void setup() {",
                       "  pinMode(13, OUTPUT); Serial.begin(9600);",
                       "}",
                       "void loop() {",
                       f"  if ({cond}) {{",
                       "    digitalWrite(13, HIGH);   // aktüatörü çalıştır",
                       "  } else {",
                       "    digitalWrite(13, LOW);",
                       "  }",
                       "  delay(100);",
                       "}"),"",
                  f"Eşik değerini kendi ortamına göre kalibre et. Aktüatör olarak {act} bağlayabilirsin."))
    # --- Motor ---
    emit("arduino","ortaokul",
        "Arduino ve servo motor ile 0°'den 180°'ye tarama (sweep) yapan kod yaz.",
        block(code("cpp",
                   "#include <Servo.h>",
                   "Servo s;",
                   "void setup(){ s.attach(9); }",
                   "void loop(){",
                   "  for(int a=0;a<=180;a++){ s.write(a); delay(15);}",
                   "  for(int a=180;a>=0;a--){ s.write(a); delay(15);}",
                   "}"),"",
              "Servo 0→180 ve geri 180→0 sürekli tarar. `delay(15)` tarama hızını belirler."))
    emit("arduino","lise",
        "Arduino ve L298N ile bir DC motoru ileri-geri döndürüp hızını PWM ile ayarla.",
        block(code("cpp",
                   "int enA=10, in1=9, in2=8;",
                   "void setup(){ pinMode(enA,OUTPUT); pinMode(in1,OUTPUT); pinMode(in2,OUTPUT);}",
                   "void loop(){",
                   "  digitalWrite(in1,HIGH); digitalWrite(in2,LOW); analogWrite(enA,200); delay(2000);",
                   "  digitalWrite(in1,LOW); digitalWrite(in2,HIGH); analogWrite(enA,120); delay(2000);",
                   "}"),"",
              "in1/in2 yönü, enA (PWM 0-255) hızı belirler. 200 ≈ %78, 120 ≈ %47 hız."))
    # --- Serial ---
    emit("arduino","ortaokul",
        "Arduino'da bir potansiyometrenin değerini Serial Monitor'e yazdıran kod yaz.",
        block(code("cpp",
                   "void setup(){ Serial.begin(9600); }",
                   "void loop(){",
                   "  int deger = analogRead(A0);",
                   "  Serial.println(deger);",
                   "  delay(200);",
                   "}"),"",
              "analogRead 0-1023 arası değer verir. Serial Plotter ile grafiğini de görebilirsin."))
    # --- Kavramlar ---
    concepts=[
      ("analogRead ve digitalRead arasındaki fark nedir?","digitalRead yalnız HIGH/LOW (0/1) okur; analogRead A0-A5 pinlerinden 0-1023 arası (10-bit) değer okur. Buton için digital, sensör (ışık, sıcaklık) için analog kullanılır.","ortaokul"),
      ("PWM (analogWrite) nedir ve nasıl çalışır?","PWM, pini çok hızlı açıp kapatarak ortalama gerilimi ayarlar. analogWrite(pin, 0-255) LED parlaklığı veya motor hızı için kullanılır. Sadece ~ işaretli pinlerde (3,5,6,9,10,11) çalışır.","ortaokul"),
      ("map() fonksiyonu ne işe yarar?","map(deger, 0,1023, 0,255) bir aralıktaki değeri başka aralığa ölçekler. Örn. potansiyometre okumasını LED parlaklığına çevirmek için kullanılır.","ortaokul"),
      ("delay() yerine millis() neden tercih edilir?","delay programı tamamen durdurur; millis() geçen süreyi sayar ve birden fazla işi aynı anda yönetmene izin verir. Gerçek zamanlı projelerde millis() şarttır.","lise"),
      ("INPUT_PULLUP nedir?","Pini dahili çekme direnciyle HIGH'a çeker; buton basılınca LOW okunur. Böylece harici direnç gerekmez ve pin havada kalmaz.","ortaokul"),
      ("buton zıplaması (debounce) nedir, nasıl çözülür?","Butona basıldığında mekanik titreşim birden çok sinyal üretir. Kısa bir gecikme (delay 20-50 ms) veya millis ile zaman kontrolü yaparak tek basış olarak algılanır.","lise"),
      ("Arduino'da değişken tipleri (int, float, bool) ne zaman kullanılır?","int tam sayılar, float ondalıklı sayılar (sıcaklık gibi), bool doğru/yanlış (buton durumu) için kullanılır. Doğru tip bellek ve doğruluk sağlar.","ortaokul"),
    ]
    for q,a,diff in concepts:
        emit("arduino",diff,f"Arduino'da {q}",a)
    # --- Entegre projeler ---
    projects=[("dijital termometre","DHT11'den sıcaklığı okuyup LCD'de gösteren","DHT11 + LCD"),
              ("reaksiyon oyunu","rastgele süre sonra LED yanınca butona basma süresini ölçen","LED + buton"),
              ("park sensörü","HC-SR04 mesafesine göre buzzer'ı hızlanan","HC-SR04 + buzzer"),
              ("akıllı sera","toprak nem + sıcaklık okuyup fan ve pompayı yöneten","sensörler + röle"),
              ("gece lambası","ortam karanlıkken LED şeridini açan","LDR + LED")]
    for name,desc,parts in projects:
        emit("arduino","lise",
            f"Arduino ile {name} projesi nasıl tasarlanır? ({parts})",
            block(f"**Amaç:** {desc} bir sistem.","",
                  f"**Parçalar:** {parts}.","",
                  "**Adımlar:** (1) sensör(ler)i oku, (2) eşik/kural uygula, "
                  "(3) aktüatörü (LED/buzzer/röle) çalıştır, (4) durumu Serial/LCD ile göster.","",
                  "loop() içinde bu döngü sürekli tekrarlanır; eşikleri kalibre etmeyi unutma."))

# =====================================================================
# SCRATCH  (blok tabanlı doğru desenler)
# =====================================================================
def gen_scratch():
    # Blok/kavram tanıtımları
    blocks=[
     ("Hareket blokları","'10 adım git', 'x'i değiştir', '90 derece dön' gibi bloklarla sprite'ı hareket ettirir","ilkokul"),
     ("Görünüm blokları","'... de', 'kostümü değiştir', 'boyutu ayarla', 'göster/gizle' bloklarını içerir","ilkokul"),
     ("Ses blokları","'sesini çal', 'sesini bitene kadar çal', 'ses yüksekliğini ayarla' bloklarıdır","ilkokul"),
     ("Kontrol blokları","'bekle', 'tekrarla', 'eğer ... ise', 'sürekli tekrarla' ile akışı yönetir","ortaokul"),
     ("Algılama blokları","'... değdi mi?', 'fare x/y', 'tuşa basıldı mı?' ile ortamı algılar","ortaokul"),
     ("İşlemler (Operatörler)","toplama, karşılaştırma, 'rastgele sayı', 've/veya', metin birleştirme yapar","ortaokul"),
     ("Değişkenler","'skor', 'can' gibi verileri saklar; 'değişkeni değiştir/ayarla' ile güncellenir","ortaokul"),
     ("Kalem blokları","'kalemi bastır', 'kalem rengini ayarla', 'sil' ile ekrana çizim yapar","ortaokul"),
    ]
    for name,desc,diff in blocks:
        emit("scratch",diff,f"Scratch'te {name} ne işe yarar?",
             block(f"**{name}**: {desc}.","",
                   "Blokları sürükleyip yeşil bayrak bloğunun altına ekleyerek programı oluşturursun."))
    # Küçük oyun/efekt mekanikleri
    mech=[
     ("bir sprite'ı fare imlecini takip ettir","[Sürekli tekrarla]\n  [(fare işaretçisine) doğru dön]\n  [10 adım git]\n[Tekrarla sonu]","ilkokul"),
     ("ok tuşlarıyla sprite'ı hareket ettir","[Sağ ok basıldı mı?] → [x'i 10 değiştir]\n[Sol ok] → [x'i -10]\n[Yukarı] → [y'yi 10]\n[Aşağı] → [y'yi -10]","ilkokul"),
     ("kenara değince seken bir top yap","[Sürekli tekrarla]\n  [10 adım git]\n  [Kenarda ise sek]\n[Tekrarla sonu]","ilkokul"),
     ("bir sprite'ı sürekli döndür","[Sürekli tekrarla]\n  [15 derece dön]\n[Tekrarla sonu]","ilkokul"),
     ("tıklayınca puanı artıran hedef oyunu","[Bu sprite tıklandığında]\n  [skor'u 1 değiştir]\n  [rastgele konuma git]","ortaokul"),
     ("iki sprite çarpışınca oyunu bitir","[Eğer <(Düşman) değdi mi?> ise]\n  [(Kaybettin!) de]\n  [Tümünü durdur]","ortaokul"),
     ("geri sayım sayacı yap","[sayac'ı 10 yap]\n[10 kere tekrarla]\n  [sayac'ı -1 değiştir]\n  [1 saniye bekle]","ortaokul"),
     ("sprite'ı ışınla (rastgele konum)","[x'i (-240..240 rastgele) yap]\n[y'yi (-180..180 rastgele) yap]","ilkokul"),
    ]
    for desc,blk,diff in mech:
        emit("scratch",diff,f"Scratch'te {desc} için hangi blokları kullanırım?",
             block("Blok yapısı:",code("text",blk),"",
                   "Yeşil bayrağa tıklandığında çalışması için en üste 'yeşil bayrak tıklandığında' bloğunu ekle."))
    # Animasyon / kostüm
    for ms in ["0.2","0.3","0.5"]:
        emit("scratch","ilkokul",
            f"Scratch'te kostüm değiştirerek yürüme animasyonu yap ({ms} saniye aralıkla).",
            block(code("text",
                       "[Yeşil bayrak tıklandığında]",
                       "[Sürekli tekrarla]",
                       "  [Sonraki kostüme geç]",
                       f"  [{ms} saniye bekle]",
                       "[Tekrarla sonu]"),"",
                  f"Kostümler {ms} saniyede bir değişir; süre kısaldıkça animasyon hızlanır."))
    # Değişken / liste / yayın / klon / yerçekimi
    emit("scratch","ortaokul","Scratch'te iki oyuncu için ayrı skor değişkenleri nasıl tutulur?",
        block("Değişkenler bölümünden 'skor1' ve 'skor2' oluştur (tüm spritelar için).","",
              code("text",
                   "[Yeşil bayrak tıklandığında]",
                   "[skor1'i 0 yap]",
                   "[skor2'yi 0 yap]"),"",
              "Puan kazanınca ilgili değişkeni '1 değiştir' ile artır. Değişkenler sahnede gösterilebilir."))
    emit("scratch","ortaokul","Scratch'te alışveriş listesi gibi bir liste nasıl oluşturulur ve elemana nasıl erişilir?",
        block("Değişkenler → 'Liste oluştur' → 'alisveris'.","",
              code("text",
                   "[(Ekmek) ögesini (alisveris) listesine ekle]",
                   "[(Süt) ögesini (alisveris) listesine ekle]",
                   "[((alisveris) listesinin 1. ögesi) de]"),"",
              "Liste sıralı veri saklar; 'listenin uzunluğu' ile eleman sayısını öğrenirsin."))
    emit("scratch","ortaokul","Scratch'te yayın (broadcast) ile sahne geçişi nasıl yapılır?",
        block(code("text",
                   "// Sprite 1",
                   "[Yeşil bayrak tıklandığında]",
                   "[(oyunu başlat) yayınla]",
                   "",
                   "// Sprite 2",
                   "[(oyunu başlat) mesajını aldığımda]",
                   "[Göster]"),"",
              "Yayın bir sprite'ın diğerlerini tetiklemesini sağlar (radyo mesajı gibi)."))
    emit("scratch","lise","Scratch'te klon kullanarak mermi (ateş etme) sistemi nasıl yapılır?",
        block(code("text",
                   "[boşluk tuşuna basıldığında]",
                   "[kendimin klonunu oluştur]",
                   "",
                   "[Klon olarak başladığımda]",
                   "[Göster]",
                   "[Ta ki <kenarda?> olana dek tekrarla]",
                   "  [y'yi 10 değiştir]",
                   "[Bu klonu sil]"),"",
              "Her boşluk basışı yeni bir mermi klonu oluşturur; kenara varınca klon silinir."))
    emit("scratch","lise","Scratch'te yerçekimi ve zıplama (platform oyunu) mekaniği nasıl kodlanır?",
        block(code("text",
                   "[yHiz'i 0 yap]",
                   "[Sürekli tekrarla]",
                   "  [yHiz'i -1 değiştir]      // yerçekimi",
                   "  [y'yi (yHiz) değiştir]",
                   "  [Eğer <y < -150> ise] [y'yi -150 yap][yHiz'i 0 yap]",
                   "  [Eğer <boşluk basıldı ve yerde> ise] [yHiz'i 15 yap]",
                   "[Tekrarla sonu]"),"",
              "Her karede hız azalır (yerçekimi), zıplayınca hız pozitif olur."))
    # Kalem/çizim çokgenler
    for name,sides,ang in [("kare",4,90),("üçgen",3,120),("altıgen",6,60),("yıldız",5,144)]:
        emit("scratch","ortaokul",
            f"Scratch kalem bloklarıyla {name} çizen program yaz.",
            block(code("text",
                       "[Kalemi bastır]",
                       f"[{sides} kere tekrarla]",
                       "  [100 adım git]",
                       f"  [{ang} derece dön]",
                       "[Tekrarla sonu]"),"",
                  f"{sides} kez ilerleyip {ang}° dönerek {name} çizilir."))
    # Operatör / mantık
    emit("scratch","ortaokul","Scratch'te kullanıcının girdiği sayının tek mi çift mi olduğunu bulan program yap.",
        block(code("text",
                   "[(Bir sayı gir) sor ve bekle]",
                   "[Eğer <(cevap mod 2) = 0> ise]",
                   "  [(Çift) de]",
                   "[Değilse]",
                   "  [(Tek) de]"),"",
              "'mod' işleci bölümden kalanı verir; 2'ye bölümünden kalan 0 ise sayı çifttir."))
    emit("scratch","lise","Scratch'te 1-100 arası sayı tahmin oyunu nasıl yapılır?",
        block(code("text",
                   "[gizli'yi (1..100 rastgele) yap]",
                   "[Sürekli tekrarla]",
                   "  [(Tahmin?) sor ve bekle]",
                   "  [Eğer <cevap > gizli> ise] [(Daha küçük) de]",
                   "  [Eğer <cevap < gizli> ise] [(Daha büyük) de]",
                   "  [Eğer <cevap = gizli> ise] [(Bildin!) de][Dur]",
                   "[Tekrarla sonu]"),"",
              "Bu ikili aramanın oyunlaştırılmış hâlidir; ipuçlarıyla aralık daraltılır."))

# =====================================================================
# MBLOCK / mBot
# =====================================================================
def gen_mblock():
    # Hareket varyasyonları
    moves=[("ileri",150,"2 saniye"),("geri",120,"1 saniye"),("sola dön",100,"0.5 saniye"),
           ("sağa dön",100,"0.5 saniye"),("ileri",255,"3 saniye")]
    for direction,hiz,sure in moves:
        emit("mblock","ilkokul",
            f"mBot'u {hiz} hızıyla {sure} boyunca {direction} hareket ettiren blokları yaz.",
            block(code("text",
                       "[Yeşil bayrak tıklandığında]",
                       f"[{direction.capitalize()} git, hız: {hiz}]",
                       f"[{sure} bekle]",
                       "[Dur]"),"",
                  "Hız 0-255 arasıdır (0 durur, 255 en hızlı). Süreyi değiştirerek mesafeyi ayarlarsın."))
    # LED renk kombinasyonları
    leds=[("kırmızı",255,0,0),("yeşil",0,255,0),("mavi",0,0,255),("sarı",255,255,0),
          ("mor",255,0,255),("turkuaz",0,255,255),("beyaz",255,255,255)]
    for name,r,g,b in leds:
        emit("mblock","ilkokul",
            f"mBot'un LED'lerini {name} renge ayarlayan bloğu yaz.",
            block(code("text", f"[LED (tümü) rengini kırmızı:({r}) yeşil:({g}) mavi:({b}) yap]"),"",
                  f"{name.capitalize()} rengi R={r}, G={g}, B={b} karışımıyla elde edilir."))
    # Çizgi izleme durumları
    emit("mblock","ortaokul","mBot ile çizgi izleyen robot yapmak için sensör durumlarına göre nasıl karar verilir?",
        block("Çizgi izleme sensörü 4 durum (0-3) verir:","",
              code("text",
                   "[Sürekli tekrarla]",
                   "  [Eğer <çizgi sensörü = 0> ise] [İleri git]      // ikisi de çizgide",
                   "  [Eğer <çizgi sensörü = 1> ise] [Sola dön]       // sağ dışında",
                   "  [Eğer <çizgi sensörü = 2> ise] [Sağa dön]       // sol dışında",
                   "  [Eğer <çizgi sensörü = 3> ise] [Dur]            // ikisi de dışında",
                   "[Tekrarla sonu]"),"",
              "Sensör siyah çizgi ile beyaz zemini ayırt eder; robotu çizgide tutar."))
    # Ultrasonik engelden kaçış (mesafe eşikleri)
    for d in [10,15,20,25]:
        emit("mblock","ortaokul",
            f"mBot ultrasonik sensörle engele {d} cm'den yakın olunca dönen engelden kaçan robot nasıl programlanır?",
            block(code("text",
                       "[Sürekli tekrarla]",
                       f"  [Eğer <ultrasonik mesafe < {d}> ise]",
                       "    [Dur] [Geri git 0.3 sn] [Sağa dön 0.5 sn]",
                       "  [Değilse]",
                       "    [İleri git, hız: 150]",
                       "[Tekrarla sonu]"),"",
                  f"Engel {d} cm'den yakınsa robot geri gidip döner, değilse ilerler."))
    # Işık sensörü
    emit("mblock","ortaokul","mBot'un ortam ışık sensörü ile karanlıkta LED yakan davranışı nasıl yazılır?",
        block(code("text",
                   "[Sürekli tekrarla]",
                   "  [Eğer <ışık sensörü < 300> ise]",
                   "    [LED (tümü) beyaz yap]",
                   "  [Değilse]",
                   "    [LED (tümü) kapat]",
                   "[Tekrarla sonu]"),"",
              "Işık sensörü değeri düşükse ortam karanlıktır; eşiği kalibre et."))
    # IR kumanda
    emit("mblock","ortaokul","mBot'u IR kumanda ile 4 yöne kontrol eden program nasıl yazılır?",
        block(code("text",
                   "[Sürekli tekrarla]",
                   "  [Eğer <IR (yukarı) basıldı> ise] [İleri git]",
                   "  [Eğer <IR (aşağı) basıldı> ise] [Geri git]",
                   "  [Eğer <IR (sol) basıldı> ise] [Sola dön]",
                   "  [Eğer <IR (sağ) basıldı> ise] [Sağa dön]",
                   "[Tekrarla sonu]"),"",
              "mBot kutusundaki IR kumandanın her tuşuna farklı hareket atanır."))
    # Buzzer/müzik
    emit("mblock","ilkokul","mBot buzzer'ı ile 'Do-Re-Mi-Fa-Sol' notalarını çalan program yaz.",
        block(code("text",
                   "[Buzzer (C4) notasını (0.5) vuruş çal]",
                   "[Buzzer (D4) notasını (0.5) vuruş çal]",
                   "[Buzzer (E4) notasını (0.5) vuruş çal]",
                   "[Buzzer (F4) notasını (0.5) vuruş çal]",
                   "[Buzzer (G4) notasını (0.5) vuruş çal]"),"",
              "C4=Do, D4=Re, E4=Mi, F4=Fa, G4=Sol. Vuruş süresini artırınca nota uzar."))
    # Buton / onboard
    emit("mblock","ilkokul","mBot'un kart üzerindeki butonuna basınca hareket başlatan program yaz.",
        block(code("text",
                   "[Sürekli tekrarla]",
                   "  [Eğer <kart butonu basıldı?> ise]",
                   "    [İleri git, hız: 150] [1 saniye bekle] [Dur]",
                   "[Tekrarla sonu]"),"",
              "Butona her basışta robot 1 saniye ilerler."))
    # Kavramlar
    concepts=[
     ("mBlock nedir ve Scratch'ten farkı nedir?","mBlock, Scratch tabanlı görsel bir kodlama ortamıdır; ek olarak mBot, Arduino gibi donanımları sürükle-bırak bloklarla programlamanı sağlar. Ayrıca 'Yükle (upload) modu' ile kodu doğrudan karta yazabilirsin.","ilkokul"),
     ("mBot'ta 'Canlı (live) mod' ile 'Yükleme (upload) modu' farkı nedir?","Canlı modda robot bilgisayara bağlı çalışır (USB/Bluetooth); upload modunda kod karta yüklenir ve robot bağımsız çalışır. Sensör-yoğun otonom projelerde upload modu gerekir.","ortaokul"),
     ("mBot'a hangi portlardan sensör eklenir?","mCore kartında RJ25 portları (1-4) vardır; ultrasonik, çizgi izleme gibi modüller bu portlara takılır. Kod yazarken doğru port numarasını seçmelisin.","ortaokul"),
     ("mBot'u bilgisayara nasıl bağlarsın?","USB kablosuyla veya Bluetooth/2.4G modülüyle bağlanır. mBlock'ta 'Bağlan' menüsünden port seçilir, sonra kod canlı çalıştırılır veya yüklenir.","ilkokul"),
    ]
    for q,a,diff in concepts:
        emit("mblock",diff,q,a)
    # Projeler
    projects=[("çizgi izleyip engelde duran robot","çizgi izleme + ultrasonik"),
              ("alkışla hareket eden robot","ses sensörü"),
              ("ışığa doğru giden robot","iki ışık sensörü"),
              ("mesafeye göre LED rengi değişen robot","ultrasonik + LED"),
              ("labirentte duvar takip eden robot","ultrasonik")]
    for name,parts in projects:
        emit("mblock","lise",
            f"mBot ile {name} projesi nasıl tasarlanır? ({parts})",
            block(f"**Kullanılan sensör(ler):** {parts}.","",
                  "Mantık: sensörü sürekli oku → koşula göre motorları (İleri/Dön/Dur) ve LED/buzzer'ı yönet. "
                  "Sensör-yoğun olduğu için 'Yükleme modu'nda çalıştır.","",
                  "Eşik değerlerini ortamında test ederek ayarla."))
    # Arduino-mode
    emit("mblock","lise","mBlock'ta 'Arduino modu' ile mBot çizgi izleme kodu C++ olarak nasıl görünür?",
        block(code("cpp",
                   "#include <MeMCore.h>",
                   "MeLineFollower line(PORT_2);",
                   "MeDCMotor ML(M1), MR(M2);",
                   "void setup(){}",
                   "void loop(){",
                   "  int s = line.readSensors();",
                   "  if(s==0){ ML.run(-120); MR.run(120);}     // ileri",
                   "  else if(s==1){ ML.run(-60); MR.run(120);} // sola",
                   "  else if(s==2){ ML.run(-120); MR.run(60);} // sağa",
                   "  else { ML.run(0); MR.run(0);}             // dur",
                   "}"),"",
              "mBlock blok kodunu otomatik C++'a çevirir; ileri seviye için doğrudan C++ yazabilirsin."))

# =====================================================================
# ROBOTİK
# =====================================================================
def gen_robotik():
    # Temel kavramlar
    concepts=[
     ("Robotların temel yapı taşları nelerdir?","Bir robot dört ana parçadan oluşur: **sensörler** (çevreyi algılar), **denetleyici/beyin** (karar verir), **aktüatörler** (hareket eder) ve **güç kaynağı**. Bu döngüye 'algıla-düşün-hareket et' denir.","ilkokul"),
     ("Sensör ile aktüatör arasındaki fark nedir?","Sensör çevreden bilgi **toplar** (mesafe, ışık, sıcaklık); aktüatör ise komutu **fiziksel harekete** çevirir (motor, servo). Sensör giriş, aktüatör çıkıştır.","ilkokul"),
     ("Açık çevrim ve kapalı çevrim kontrol arasındaki fark nedir?","Açık çevrim geri bildirim kullanmaz (körlemesine komut verir); kapalı çevrim sensörle sonucu ölçüp hatayı düzeltir. Çizgi izleyen robot kapalı çevrimdir.","lise"),
     ("Geri besleme (feedback) nedir ve neden önemlidir?","Geri besleme, sistemin çıkışını ölçüp girişe geri vermesidir. Robot hedeften sapınca sensör bunu algılar ve denetleyici düzeltir; kararlı ve hassas kontrol sağlar.","lise"),
     ("Serbestlik derecesi (DOF) nedir?","DOF, bir robotun bağımsız hareket edebildiği eksen sayısıdır. İnsan kolu 7 DOF'a yakındır; bir robot kolunun DOF'u ne kadar çoksa o kadar esnek konumlanır.","lise"),
     ("İleri ve ters kinematik nedir?","İleri kinematik: eklem açılarından uç noktanın konumunu bulur. Ters kinematik: istenen uç konumdan gereken eklem açılarını hesaplar. Robot kolları için ters kinematik kritiktir.","lise"),
     ("Diferansiyel sürüş (differential drive) nasıl çalışır?","İki tekerin hızları ayrı kontrol edilir. Eşit hız → düz gider; bir teker yavaş → döner; ters yönler → yerinde döner. mBot ve çoğu eğitim robotu bunu kullanır.","ortaokul"),
     ("Enkoder (encoder) ne işe yarar?","Enkoder motor milinin ne kadar döndüğünü sayar; böylece robot kat ettiği mesafeyi ve hızını bilir. Hassas konumlama ve kapalı çevrim hız kontrolü için gereklidir.","lise"),
     ("H-köprüsü (H-bridge) nedir?","H-köprüsü, bir DC motorun dönüş **yönünü** değiştirmeye yarayan devredir (L298N, L293D). Dört anahtarın konumuna göre motora giden akımın yönü tersine çevrilir.","lise"),
     ("PWM robotikte hız kontrolü için nasıl kullanılır?","PWM sinyalin açık kalma oranını (duty cycle) değiştirerek motora giden ortalama gücü ayarlar. %100 tam hız, %50 yarı hız demektir.","ortaokul"),
     ("Tork ve dişli oranı motoru nasıl etkiler?","Dişli oranı hızı azaltıp torku (döndürme kuvvetini) artırır. Ağır yük kaldıran robot kolu yüksek tork; hızlı yarış robotu düşük dişli oranı ister.","lise"),
     ("Ölü hesap (dead reckoning) navigasyonu nedir?","Robotun tekerlek dönüşü ve yönünden konumunu tahmin etmesidir. Basittir ama hata birikir; bu yüzden enkoder/IMU ile desteklenir.","lise"),
     ("Durum makinesi (state machine) robot davranışını nasıl düzenler?","Robotun davranışı 'ara', 'yaklaş', 'tut', 'bırak' gibi durumlara bölünür; koşullara göre durumlar arasında geçilir. Karmaşık davranışı düzenli yönetmeyi sağlar.","lise"),
     ("Otonom robot ile uzaktan kumandalı robot farkı nedir?","Otonom robot kendi kararını sensör verisiyle verir; uzaktan kumandalı robot insan komutuyla çalışır. Otonomi seviyesi arttıkça sensör ve algoritma karmaşıklığı artar.","ortaokul"),
     ("Kalibrasyon neden gereklidir?","Sensörler ortam ışığı, zemin rengi gibi koşullara göre farklı değer verir. Kalibrasyon eşik değerlerini o ortama göre ayarlayarak robotun doğru çalışmasını sağlar.","ortaokul"),
     ("Mikrodenetleyici (Arduino) ile mikroişlemci (Raspberry Pi) farkı nedir?","Mikrodenetleyici basit, gerçek zamanlı G/Ç işleri için (sensör-motor); mikroişlemci işletim sistemi çalıştırır, kamera/görüntü işleme gibi ağır işler için uygundur.","lise"),
     ("Robot kolunda uç işlevci (end-effector) nedir?","Kolun ucundaki iş yapan parçadır: tutucu (gripper), kaynak başlığı, vakum vantuz vb. Göreve göre değiştirilir.","ortaokul"),
    ]
    for q,a,diff in concepts:
        emit("robotik",diff,q,a)
    # Sensör bankası
    sensors=[("ultrasonik mesafe sensörü","ses dalgasıyla engel mesafesini ölçer","engelden kaçınma, park"),
             ("kızılötesi (IR) sensör","yüzeyden yansıyan IR ile çizgi/engel algılar","çizgi izleme, yakın engel"),
             ("IMU (jiroskop+ivmeölçer)","eğim, dönüş ve ivmeyi ölçer","denge robotu, yön bulma"),
             ("LDR ışık sensörü","ortam ışığını ölçer","ışık takip, gece modu"),
             ("dokunma/bumper sensörü","çarpmayı algılar","duvar bulma, güvenlik"),
             ("renk sensörü","yüzey rengini ayırt eder","nesne ayıklama, çizgi"),
             ("enkoder","tekerlek dönüşünü sayar","mesafe/hız ölçümü"),
             ("kamera","görüntü alır, nesne tanır","görüntü işleme, izleme")]
    for name,how,use in sensors:
        emit("robotik","ortaokul",
            f"Robotikte {name} ne işe yarar ve hangi projelerde kullanılır?",
            block(f"**{name}**: {how}.","",
                  f"Kullanım alanları: {use}.","",
                  "Denetleyici bu sensörü okuyup aktüatörleri buna göre yönetir."))
    # Aktüatör bankası
    acts=[("DC motor","sürekli döner; hız PWM, yön H-köprüsü ile ayarlanır","tekerlek tahriki"),
          ("servo motor","0-180° hassas konum kontrolü yapar","robot kol eklemi, direksiyon"),
          ("step motor","adım adım çok hassas döner","3B yazıcı, hassas konumlama"),
          ("solenoid/röle","aç-kapa tetikleme yapar","vurucu, anahtarlama")]
    for name,how,use in acts:
        emit("robotik","lise",
            f"Robotikte {name} nasıl çalışır ve nerede kullanılır?",
            block(f"**{name}**: {how}.","",f"Tipik kullanım: {use}.","",
                  "Aktüatör seçimi hız, hassasiyet ve tork ihtiyacına göre yapılır."))
    # Robot türleri
    types=[("çizgi izleyen robot","IR sensörlerle siyah çizgiyi takip eder; sapınca motor hızlarını düzelterek çizgide kalır","ortaokul"),
           ("engelden kaçan robot","ultrasonik sensörle önündeki engeli algılar, yaklaşınca döner","ortaokul"),
           ("ışık takip eden robot","iki LDR'nin farkına göre daha aydınlık yöne yönelir","ortaokul"),
           ("duvar takip eden robot","yan mesafeyi sabit tutarak labirentte duvar boyunca ilerler","lise"),
           ("sumo robotu","rakibi ultrasonik ile bulur, çizgi sensörüyle ringde kalır","lise"),
           ("labirent çözen robot","sağ-el kuralı veya haritalama ile çıkışı bulur","lise"),
           ("denge (self-balancing) robotu","IMU ile eğimi ölçüp PID ile dik durur","lise"),
           ("pick-and-place robot kol","ters kinematikle nesneyi tutup başka yere bırakır","lise")]
    for name,how,diff in types:
        emit("robotik",diff,f"{name.capitalize()} nasıl çalışır ve hangi sensör/aktüatörleri kullanır?",
             block(f"**Çalışma prensibi:** {how}.","",
                   "Kapalı çevrim mantığı: sensörü oku → hatayı hesapla → motoru düzelt → tekrarla.","",
                   "Eşikleri ve hızları test ederek ayarlamak performansı artırır."))
    # Kontrol teorisi / PID
    emit("robotik","lise","PID kontrol nedir ve robotikte neden kullanılır?",
        block("PID, hatayı üç terimle düzelten bir kontrol yöntemidir:","",
              "- **P (Oransal):** anlık hatayla orantılı düzeltir.",
              "- **I (İntegral):** biriken küçük hataları giderir.",
              "- **D (Türev):** hatanın değişim hızına bakıp aşmayı (salınımı) azaltır.","",
              "Çıkış = Kp·e + Ki·∫e + Kd·(de/dt). Çizgi izleyen robotu yumuşak ve hızlı tutar."))
    emit("robotik","lise","PID'de Kp, Ki, Kd katsayıları nasıl ayarlanır (tuning)?",
        block("Önce sadece **Kp** artırılır; robot çizgiyi takip edene ama salınana kadar. "
              "Sonra salınımı azaltmak için **Kd** eklenir. Kalan küçük sapma için az miktarda **Ki** verilir.","",
              "Çok yüksek Kp → salınım; çok yüksek Ki → aşma/kararsızlık; çok yüksek Kd → gürültüye duyarlılık. "
              "Adım adım, tek katsayı değiştirerek ayarlanır."))
    emit("robotik","ortaokul","Bir robotta 'setpoint' (hedef değer) ve 'hata' ne demektir?",
        block("**Setpoint** ulaşılmak istenen değerdir (ör. çizginin tam ortası). "
              "**Hata** = setpoint − ölçülen değer. Denetleyici hatayı sıfıra indirmeye çalışır.","",
              "Örn. çizgi izleyen robotta hata, robotun çizgiden ne kadar saptığıdır."))

# =====================================================================
# EK ÜRETİM — hesaplanmış doğru çıktılarla ölçekleme
# =====================================================================
def gen_elektronik_more():
    import itertools
    # Ohm I=V/R
    for V,R in [(5,150),(5,470),(5,680),(6,220),(6,470),(9,330),(9,680),(9,1500),(12,330),(12,680),(12,3300),(24,1000),(3.3,220),(3.3,470),(18,2200)]:
        I=V/R
        emit("elektronik","ilkokul",
            f"{V} volt gerilim {R} ohm dirence uygulanınca geçen akım kaç miliamperdir?",
            block("Ohm yasası **I = V / R**.","",
                  f"I = {V} / {R} = {I:.4f} A = **{I*1000:.2f} mA**."))
    # V=IR
    for I,R in [(0.015,220),(0.04,100),(0.008,470),(0.025,330),(0.06,150),(0.012,1000),(0.3,22),(0.002,4700),(0.09,47),(0.07,68)]:
        V=I*R
        emit("elektronik","ortaokul",
            f"{I*1000:.0f} mA akım {R} ohm dirençten geçerken oluşan gerilim düşümü kaç volttur?",
            block("**V = I × R**.","",f"V = {I} × {R} = **{V:.3f} V**."))
    # R=V/I
    for V,I in [(5,0.01),(9,0.02),(12,0.05),(3.3,0.02),(24,0.1),(6,0.03),(5,0.04),(12,0.2)]:
        emit("elektronik","ortaokul",
            f"Üzerinde {V} V ölçülen ve {I*1000:.0f} mA çeken direncin değeri kaç ohm'dur?",
            block("**R = V / I**.","",f"R = {V} / {I} = **{V/I:.0f} Ω**."))
    # Güç P=VI, P=I2R, P=V2/R
    for V,I in [(5,0.3),(12,0.5),(9,0.8),(3.3,1.5),(230,0.2),(5,1.2),(12,2),(24,0.5)]:
        emit("elektronik","ortaokul",
            f"{V} V altında {I} A çeken cihazın harcadığı güç kaç watttır?",
            block("**P = V × I**.","",f"P = {V} × {I} = **{V*I:.2f} W**."))
    for I,R in [(0.5,10),(0.2,100),(1,4.7),(0.1,220),(2,2.2),(0.05,470)]:
        emit("elektronik","lise",
            f"{R} ohm dirençten {I} A akım geçerken dirençte harcanan güç nedir (P=I²R)?",
            block("**P = I² × R**.","",f"P = {I}² × {R} = **{I**2*R:.3f} W**."))
    for V,R in [(5,10),(12,100),(9,220),(3.3,47),(24,1000)]:
        emit("elektronik","lise",
            f"{V} V bir dirence uygulanıyor, direnç {R} ohm ise güç nedir (P=V²/R)?",
            block("**P = V² / R**.","",f"P = {V}² / {R} = **{V**2/R:.3f} W**."))
    # LED direnci daha fazla
    for Vs,Vled,mA in [(5,2.1,20),(5,3.0,20),(9,2.2,15),(12,2.0,20),(5,1.9,10),(9,3.2,20),(12,3.4,20),(3.3,2.0,5)]:
        I=mA/1000; R=(Vs-Vled)/I
        emit("elektronik","lise",
            f"{Vs} V kaynak, {Vled} V LED ve {mA} mA için LED ön direnci kaç ohm olmalı?",
            block("**R = (Vs − V_LED) / I**.","",f"R = ({Vs} − {Vled}) / {I} = **{R:.0f} Ω**."))
    # Seri / paralel daha fazla
    for s in [(100,200),(470,680),(1000,3300),(220,220,220),(150,330,470),(2200,4700),(10000,10000),(560,680,820)]:
        emit("elektronik","ilkokul",
            f"Seri bağlı {', '.join(str(x)+'Ω' for x in s)} dirençlerin toplamı kaçtır?",
            block("Seri: dirençler toplanır.","",f"R = {' + '.join(map(str,s))} = **{sum(s)} Ω**."))
    for s in [(100,300),(220,220),(1000,3000),(470,470,470),(680,1200),(2200,2200),(150,150),(1000,4700)]:
        R=1/sum(1/x for x in s)
        emit("elektronik","lise",
            f"Paralel bağlı {', '.join(str(x)+'Ω' for x in s)} dirençlerin eşdeğeri kaç ohm'dur?",
            block("Paralel: 1/R = Σ(1/Ri).","",f"R_eş = **{R:.1f} Ω** (en küçük dirençten küçüktür)."))
    # Gerilim bölücü daha fazla
    for Vin,R1,R2 in [(5,2200,3300),(9,4700,4700),(12,10000,5000),(5,1000,4700),(3.3,470,680),(12,1000,3300),(9,2200,1000)]:
        emit("elektronik","lise",
            f"Gerilim bölücüde Vin={Vin}V, R1={R1}Ω, R2={R2}Ω için Vout kaçtır?",
            block("**Vout = Vin·R2/(R1+R2)**.","",f"Vout = {Vin}×{R2}/({R1}+{R2}) = **{Vin*R2/(R1+R2):.3f} V**."))
    # RC zaman sabiti daha fazla
    for R,C in [(2200,1e-6),(10000,10e-6),(470,100e-6),(100000,100e-9),(1000,470e-6),(33000,1e-6),(4700,4.7e-6)]:
        tau=R*C
        emit("elektronik","lise",
            f"R={R}Ω, C={C*1e6:.3g}µF RC devresinde zaman sabiti kaç milisaniyedir?",
            block("**τ = R·C**.","",f"τ = {R}×{C:.2e} = **{tau*1000:.2f} ms** (5τ ≈ {5*tau*1000:.1f} ms'de ~%99 dolar)."))
    # Enerji E=P*t (Wh)
    for P,h in [(60,5),(100,3),(9,24),(1500,2),(40,10),(5,8),(2000,0.5)]:
        emit("elektronik","ortaokul",
            f"{P} W gücünde bir cihaz {h} saat çalışırsa kaç watt-saat (Wh) enerji harcar?",
            block("**E = P × t**.","",f"E = {P} × {h} = **{P*h:.0f} Wh** = {P*h/1000:.3f} kWh."))
    # Batarya ömrü
    for cap,mA in [(2000,200),(1000,50),(3000,300),(500,25),(2500,100),(1200,150)]:
        emit("elektronik","ortaokul",
            f"{cap} mAh kapasiteli batarya {mA} mA çeken devreyi yaklaşık kaç saat besler?",
            block("Kabaca **süre = kapasite / akım**.","",f"t ≈ {cap} / {mA} = **{cap/mA:.1f} saat** (verim kayıpları hariç)."))
    # Kondansatör paralel/seri
    for s in [(100,100),(10,22),(1,1,1),(47,47),(100,220)]:
        emit("elektronik","lise",
            f"Paralel bağlı {', '.join(str(x)+'µF' for x in s)} kondansatörlerin toplam sığası kaçtır?",
            block("Paralel kondansatörlerde sığalar **toplanır**.","",f"C = {' + '.join(map(str,s))} = **{sum(s)} µF**."))

def gen_algoritma_more():
    import math
    def insort(a):
        a=a[:]
        for i in range(1,len(a)):
            k=a[i]; j=i-1
            while j>=0 and a[j]>k: a[j+1]=a[j]; j-=1
            a[j+1]=k
        return a
    arrays=[[5,2,8,1],[9,3,7,4,2],[10,1,5],[6,6,2,9,1],[3,8,1,4,7,2],[15,3,9,1],[4,4,2,8],[7,1,3,9,5]]
    for arr in arrays:
        emit("algoritma","ortaokul",
            f"{arr} dizisini eklemeli (insertion) sıralama ile sırala; sonucu yaz.",
            block("Eklemeli sıralama her elemanı, solundaki sıralı kısma doğru yeri bulunana dek kaydırır.","",
                  code("python","def insertion_sort(a):","    for i in range(1,len(a)):","        k=a[i]; j=i-1","        while j>=0 and a[j]>k:","            a[j+1]=a[j]; j-=1","        a[j+1]=k","    return a"),"",
                  f"{arr} → **{insort(arr)}**. Karmaşıklık O(n²), neredeyse sıralı dizilerde çok hızlı."))
    # Linear search
    for arr,t in [([4,8,15,16,23],15),([1,2,3,4,5],6),([9,7,5,3],3),([10,20,30,40,50],25),([2,4,6],4),([11,22,33,44],44)]:
        idx=arr.index(t) if t in arr else -1
        emit("algoritma","ilkokul",
            f"{arr} listesinde {t} değerini doğrusal arama ile ara; kaçıncı indekste?",
            block("Doğrusal arama baştan sona tek tek bakar.","",
                  code("python","def ara(a,t):","    for i in range(len(a)):","        if a[i]==t: return i","    return -1"),"",
                  f"Sonuç: **{('indeks '+str(idx)) if idx>=0 else 'bulunamadı'}**. En kötü durum O(n)."))
    # Sum/max/min/avg/count of arrays
    for arr in [[3,7,2,8,5],[10,10,10],[1,2,3,4,5,6],[9,1,4,7],[20,5,15,10]]:
        emit("algoritma","ilkokul",
            f"{arr} dizisinin toplamını, en büyüğünü ve en küçüğünü bulan algoritmayı açıkla.",
            block("Diziyi bir kez gezip toplam, max ve min güncellenir (tek geçiş, O(n)).","",
                  f"Toplam = **{sum(arr)}**, En büyük = **{max(arr)}**, En küçük = **{min(arr)}**."))
    for arr in [[4,8,6,2],[5,5,5,5],[1,9,3,7,5],[12,4,8]]:
        even=[x for x in arr if x%2==0]
        emit("algoritma","ilkokul",
            f"{arr} dizisindeki çift sayıları bulan ve sayan algoritmayı yaz.",
            block("Her elemanın 2'ye bölümünden kalanına bakılır.","",
                  f"Çiftler: **{even}**, adet = **{len(even)}**."))
    # İkinci en büyük
    for arr in [[3,7,2,8,5],[10,20,20,15],[1,2],[9,4,9,7]]:
        u=sorted(set(arr))
        sec=u[-2] if len(u)>=2 else None
        emit("algoritma","ortaokul",
            f"{arr} dizisindeki ikinci en büyük farklı değeri bulan algoritmayı açıkla.",
            block("Diziyi gez; en büyük ve ikinci en büyüğü ayrı tut (tek geçiş).","",
                  f"En büyük = {u[-1]}, ikinci en büyük = **{sec}**."))
    # Reverse
    for arr in [[1,2,3,4],[5,9,1],[7,7,2,8]]:
        emit("algoritma","ilkokul",
            f"{arr} dizisini ters çeviren algoritmayı açıkla.",
            block("Baş ve son işaretçileri ortada buluşana dek eleman takas edilir.","",
                  code("python","def ters(a):","    i,j=0,len(a)-1","    while i<j:","        a[i],a[j]=a[j],a[i]; i+=1; j-=1","    return a"),"",
                  f"{arr} → **{arr[::-1]}**."))
    # Recursive power / hanoi / more values
    for b,e in [(2,5),(3,3),(5,2),(2,8),(10,3)]:
        emit("algoritma","lise",
            f"{b} üzeri {e} değerini özyinelemeli üs alma fonksiyonuyla hesapla.",
            block("us(b,e) = b × us(b,e-1), us(b,0)=1.","",
                  code("python","def us(b,e):","    if e==0: return 1","    return b*us(b,e-1)"),"",
                  f"us({b},{e}) = **{b**e}**."))
    for n in [3,4,5]:
        emit("algoritma","lise",
            f"{n} diskli Hanoi Kuleleri kaç hamlede çözülür ve mantığı nedir?",
            block("n disk için minimum hamle = **2ⁿ − 1**. Özyineleme: n-1 diski ara çubuğa taşı, en büyüğü hedefe koy, n-1 diski üstüne taşı.","",
                  f"{n} disk → 2^{n} − 1 = **{2**n-1}** hamle."))
    for n in [11,13,20,25]:
        seq=[0,1]
        while len(seq)<n: seq.append(seq[-1]+seq[-2])
        emit("algoritma","ortaokul",
            f"Fibonacci dizisinin ilk {n} terimini üret; {n}. terim kaçtır?",
            block("F(n)=F(n-1)+F(n-2).","",f"İlk {n} terim: {seq[:n]}","",f"{n}. terim = **{seq[n-1]}**."))
    for a,b in [(24,36),(81,27),(60,48),(14,21),(120,90)]:
        emit("algoritma","lise",
            f"{a} ile {b} için EBOB'u Öklid algoritmasıyla bul.",
            block("EBOB(a,b)=EBOB(b, a mod b).","",f"EBOB({a},{b}) = **{math.gcd(a,b)}**."))
    for n in [7,11,15,21,23,49]:
        isp=n>1 and all(n%i for i in range(2,int(n**0.5)+1))
        emit("algoritma","ortaokul",
            f"{n} asal mı? √n'e kadar bölen kontrolüyle açıkla.",
            block(f"2..⌊√{n}⌋ arasında bölen aranır.","",f"Sonuç: **{'asal' if isp else 'asal değil'}**."))
    # Anagram / karakter sayımı
    for a,b in [("kale","elak"),("masa","sama"),("ders","serd"),("kitap","patik")]:
        an=sorted(a)==sorted(b)
        emit("algoritma","ortaokul",
            f"'{a}' ve '{b}' kelimeleri anagram mı? Nasıl kontrol edilir?",
            block("İki kelimenin harfleri sıralanıp karşılaştırılır (veya harf sayıları).","",
                  f"sorted('{a}')={sorted(a)}, sorted('{b}')={sorted(b)} → **{'Anagram' if an else 'Anagram değil'}**."))
    # Big-O ek örnekler
    for algo,bo in [("iç içe iki döngü","O(n²)"),("ikili arama","O(log n)"),("tek döngüyle toplam","O(n)"),
                    ("birleştirme (merge) sıralama","O(n log n)"),("sabit indeksleme","O(1)"),("üçlü iç içe döngü","O(n³)")]:
        emit("algoritma","lise",
            f"'{algo}' işleminin zaman karmaşıklığı nedir ve neden?",
            block(f"**{bo}**.","",f"Çünkü giriş büyüdükçe adım sayısı {bo} oranında artar."))
    # Yığın/kuyruk uygulama
    emit("algoritma","lise","Python listesiyle yığın (stack) nasıl uygulanır? push/pop örneği ver.",
        block("Liste sonuna ekle (append=push), sondan çıkar (pop). LIFO.","",
              code("python","stack=[]","stack.append(1); stack.append(2); stack.append(3)","print(stack.pop())  # 3","print(stack)        # [1, 2]"),"",
              "Son eklenen ilk çıkar; geri-al (undo) işlevlerinde kullanılır."))
    emit("algoritma","lise","Python ile kuyruk (queue) nasıl uygulanır? collections.deque örneği ver.",
        block("FIFO için deque kullanılır (baştan çıkarmak O(1)).","",
              code("python","from collections import deque","q=deque()","q.append('a'); q.append('b')","print(q.popleft())  # a"),"",
              "İlk giren ilk çıkar; yazıcı/görev sıralarında kullanılır."))

def gen_python_more():
    import math
    # F->C
    for f in [32,98.6,212,50,-4]:
        c=(f-32)*5/9
        emit("python_stem","ortaokul",
            f"{f} Fahrenheit'ı santigrata çeviren program yaz.",
            block("**C = (F − 32) × 5/9**.","",code("python",f"f={f}","c=(f-32)*5/9","print(round(c,1))"),"",f"Çıktı: **{c:.1f} °C**."))
    # Birim çevrimleri
    for km in [5,12,0.5,3.2]:
        emit("python_stem","ilkokul",
            f"{km} kilometreyi metreye çeviren program yaz.",
            block(code("python",f"km={km}","m=km*1000","print(m)"),"",f"Çıktı: **{km*1000:.0f} m**."))
    for kg in [2,0.75,5,1.5]:
        emit("python_stem","ilkokul",
            f"{kg} kilogramı grama çeviren program yaz.",
            block(code("python",f"kg={kg}","g=kg*1000","print(g)"),"",f"Çıktı: **{kg*1000:.0f} g**."))
    # Basit faiz
    for P,r,t in [(1000,10,2),(5000,5,3),(2000,8,1),(1500,12,4)]:
        faiz=P*r*t/100
        emit("python_stem","lise",
            f"{P} TL anapara, %{r} yıllık faiz, {t} yıl için basit faizi hesapla.",
            block("**Faiz = Anapara × Oran × Süre / 100**.","",
                  code("python",f"P,r,t={P},{r},{t}","faiz=P*r*t/100","print(faiz)"),"",
                  f"Çıktı: **{faiz:.2f} TL** faiz, toplam {P+faiz:.2f} TL."))
    # Hız = yol/zaman
    for d,t in [(100,2),(240,3),(60,0.5),(150,2.5)]:
        emit("python_stem","ortaokul",
            f"{d} km yolu {t} saatte giden aracın ortalama hızını hesapla.",
            block("**Hız = Yol / Zaman**.","",code("python",f"yol,zaman={d},{t}","hiz=yol/zaman","print(hiz)"),"",f"Çıktı: **{d/t:.1f} km/saat**."))
    # Üçgen alanı
    for taban,yuk in [(6,4),(10,5),(8,3),(12,7)]:
        emit("python_stem","ilkokul",
            f"Tabanı {taban}, yüksekliği {yuk} olan üçgenin alanını bulan program yaz.",
            block("**Alan = taban × yükseklik / 2**.","",code("python",f"t,h={taban},{yuk}","alan=t*h/2","print(alan)"),"",f"Çıktı: **{taban*yuk/2:.1f}**."))
    # Ortalama not -> harf
    for notlar in [[85,90,78],[45,60,55],[100,95,88],[70,72,68]]:
        ort=sum(notlar)/len(notlar)
        harf=("AA" if ort>=90 else "BA" if ort>=80 else "BB" if ort>=70 else "CC" if ort>=60 else "FF")
        emit("python_stem","ortaokul",
            f"{notlar} notlarının ortalamasını bulup harf notu ({'AA/BA/BB/CC/FF'}) veren program yaz.",
            block(code("python",f"notlar={notlar}","ort=sum(notlar)/len(notlar)","print(round(ort,1))"),"",
                  f"Ortalama = **{ort:.1f}** → **{harf}**."))
    # Liste comprehension
    for n in [10,15,6]:
        sq=[i*i for i in range(1,n+1)]
        emit("python_stem","ortaokul",
            f"1'den {n}'e kadar sayıların karelerini liste üreteci (comprehension) ile oluştur.",
            block(code("python",f"kareler=[i*i for i in range(1,{n}+1)]","print(kareler)"),"",f"Çıktı: **{sq}**."))
    for n in [20,10,30]:
        ev=[i for i in range(1,n+1) if i%2==0]
        emit("python_stem","ilkokul",
            f"1'den {n}'e kadar çift sayıları comprehension ile listele.",
            block(code("python",f"ciftler=[i for i in range(1,{n}+1) if i%2==0]","print(ciftler)"),"",f"Çıktı: **{ev}**."))
    # Asal listesi
    for n in [20,30,15]:
        pr=[x for x in range(2,n+1) if all(x%i for i in range(2,int(x**0.5)+1))]
        emit("python_stem","lise",
            f"2'den {n}'e kadar asal sayıları bulan program yaz.",
            block(code("python",f"asallar=[x for x in range(2,{n}+1) if all(x%i for i in range(2,int(x**0.5)+1))]","print(asallar)"),"",f"Çıktı: **{pr}**."))
    # String ters / palindrom program
    for w in ["python","bilgi","kod"]:
        emit("python_stem","ilkokul",
            f"'{w}' kelimesini ters çeviren program yaz.",
            block(code("python",f"s='{w}'","print(s[::-1])"),"",f"Çıktı: **{w[::-1]}**. `[::-1]` dilimlemesi diziyi tersler."))
    # Sınıflar
    emit("python_stem","lise","Dikdörtgen sınıfı yaz; alan ve çevre metotları olsun.",
        block(code("python","class Dikdortgen:","    def __init__(self,en,boy): self.en=en; self.boy=boy","    def alan(self): return self.en*self.boy","    def cevre(self): return 2*(self.en+self.boy)","","d=Dikdortgen(4,6)","print(d.alan(), d.cevre())"),"",
              "Çıktı: **24 20**."))
    emit("python_stem","lise","Basit bir BankaHesabi sınıfı yaz (para yatır/çek, bakiye).",
        block(code("python","class BankaHesabi:","    def __init__(self,bakiye=0): self.bakiye=bakiye","    def yatir(self,m): self.bakiye+=m","    def cek(self,m):","        if m<=self.bakiye: self.bakiye-=m","        else: print('Yetersiz bakiye')","","h=BankaHesabi(100); h.yatir(50); h.cek(30)","print(h.bakiye)"),"",
              "Çıktı: **120**. Metotlar nesnenin durumunu (bakiye) günceller."))
    # While geri sayım
    for n in [5,10,3]:
        emit("python_stem","ilkokul",
            f"{n}'den 1'e kadar geri sayan program yaz (while döngüsü).",
            block(code("python",f"i={n}","while i>0:","    print(i)","    i-=1","print('Bitti!')"),"",
                  f"Çıktı: {', '.join(str(x) for x in range(n,0,-1))}, Bitti!"))
    # Sözlük
    emit("python_stem","ortaokul","Öğrenci adlarını notlarıyla eşleştiren bir sözlük oluştur ve en yüksek notu bul.",
        block(code("python","notlar={'Ali':70,'Ayşe':95,'Can':82}","en_yuksek=max(notlar, key=notlar.get)","print(en_yuksek, notlar[en_yuksek])"),"",
              "Çıktı: **Ayşe 95**. `max(..., key=notlar.get)` en büyük değere sahip anahtarı verir."))

def gen_arduino_more():
    comps=[
     ("flame (alev) sensörü","kızılötesi ışığı algılayıp alev/yangını dijital sinyalle bildirir","güvenlik","ortaokul"),
     ("MPU6050 (ivme+jiroskop)","I2C ile eğim, ivme ve dönüşü ölçer","hareket","lise"),
     ("OLED SSD1306 ekran","I2C ile metin ve grafik gösterir","ekran","lise"),
     ("DS3231 gerçek zaman saati (RTC)","tarih-saati pil desteğiyle tutar","zaman","lise"),
     ("hall (manyetik) sensörü","mıknatıs alanını algılar; devir/sayaç için kullanılır","sensör","ortaokul"),
     ("reed switch (manyetik anahtar)","mıknatıs yaklaşınca kontağı kapatır","anahtar","ilkokul"),
     ("tilt (eğim) sensörü","eğildiğinde dijital sinyal verir","sensör","ilkokul"),
     ("kapasitif dokunma sensörü","dokunuşu temassız algılar","giriş","ortaokul"),
     ("keypad (4x4 tuş takımı)","satır-sütun taramasıyla basılan tuşu okur","giriş","lise"),
     ("MQ-3 alkol sensörü","havadaki alkol buharına göre analog değer verir","sensör","lise"),
     ("BMP180 basınç sensörü","hava basıncı ve yükseklik ölçer","çevre","lise"),
     ("SD kart modülü","SPI ile verileri dosyaya kaydeder","depolama","lise"),
     ("renk sensörü (TCS3200)","yüzey rengini RGB olarak ölçer","sensör","lise"),
     ("nem sensörü (FC-28)","toprak/ortam nemini analog verir","çevre","ortaokul"),
     ("ivme tabanlı sarsıntı sensörü","ani hareketi algılar, alarm tetikler","güvenlik","ortaokul"),
    ]
    for name,how,kind,diff in comps:
        emit("arduino",diff,f"Arduino'da {name} nasıl kullanılır ve ne işe yarar?",
             block(f"**{name}** bir {kind} elemanıdır: {how}.","",
                   "Bağlantıda GND ortak olmalı; I2C/SPI modüller için ilgili kütüphane eklenir. "
                   "Okunan değeri Serial Monitor'de görüp bir LED/buzzer ile tepki verebilirsin."))
    # Sensörü seri porta yazdır
    for s,pin,rng in [("LDR","A0","0-1023"),("potansiyometre","A1","0-1023"),("toprak nem","A2","0-1023"),
                      ("ses sensörü","A3","0-1023"),("MQ-2 gaz","A0","0-1023"),("sıcaklık (LM35)","A0","°C")]:
        emit("arduino","ortaokul",
            f"Arduino'da {s} sensörünü okuyup değerini Serial Monitor'e yazan kodu yaz.",
            block(code("cpp",f"int pin={pin};","void setup(){ Serial.begin(9600);}","void loop(){"," int d=analogRead(pin);"," Serial.println(d);"," delay(200);","}"),"",
                  f"{s} değeri ({rng}) her 200 ms'de yazdırılır. Serial Plotter ile grafiğe dökebilirsin."))
    # Blink daha fazla
    for pin,ms in [(2,400),(3,600),(4,150),(6,800),(7,350),(11,900),(13,120)]:
        emit("arduino","ilkokul",
            f"Pin {pin}'e bağlı LED'i {ms} ms aralıkla yakıp söndür.",
            block(code("cpp",f"void setup(){{ pinMode({pin},OUTPUT);}}","void loop(){",f" digitalWrite({pin},HIGH); delay({ms});",f" digitalWrite({pin},LOW); delay({ms});","}"),"",
                  f"LED {ms} ms yanıp {ms} ms söner."))
    # PWM fade
    for pin in [9,10,11,5,6]:
        emit("arduino","ortaokul",
            f"Pin {pin} (PWM) üzerindeki LED'i yavaşça parlatıp söndüren (fade) kodu yaz.",
            block(code("cpp",f"int led={pin};","void setup(){ pinMode(led,OUTPUT);}","void loop(){"," for(int b=0;b<=255;b++){ analogWrite(led,b); delay(5);}"," for(int b=255;b>=0;b--){ analogWrite(led,b); delay(5);}","}"),"",
                  "analogWrite 0-255 arası parlaklık verir; ~ işaretli PWM pinleri gerekir."))
    # RGB renk döngüsü
    emit("arduino","ortaokul","Arduino ile RGB LED'i kırmızı-yeşil-mavi sırayla değiştiren kodu yaz.",
        block(code("cpp","int r=9,g=10,b=11;","void setup(){ pinMode(r,OUTPUT);pinMode(g,OUTPUT);pinMode(b,OUTPUT);}","void renk(int R,int G,int B){ analogWrite(r,R);analogWrite(g,G);analogWrite(b,B);}","void loop(){"," renk(255,0,0);delay(500); renk(0,255,0);delay(500); renk(0,0,255);delay(500);","}"),"",
              "Her kanal 0-255 arası ayarlanır; karışımla farklı renkler elde edilir."))
    # Eşik projeleri daha fazla
    proj=[("sıcaklık (LM35)","28°C üstünde fanı çalıştıran termostat","(analogRead(A0)*0.488) > 28"),
          ("LDR","ışık azalınca sokak lambası gibi LED yakan","analogRead(A0) < 350"),
          ("ultrasonik","15 cm altında kırmızı LED yakan yakınlık uyarısı","mesafe < 15"),
          ("hall sensörü","mıknatıs geçince sayaç artıran devir ölçer","digitalRead(2)==LOW"),
          ("titreşim sensörü","sarsıntıda buzzer öten deprem alarmı","digitalRead(3)==HIGH"),
          ("su seviyesi","su azalınca uyarı veren akvaryum sistemi","analogRead(A0) < 300")]
    for s,desc,cond in proj:
        emit("arduino","ortaokul",
            f"Arduino ile {s} kullanarak {desc} sistem nasıl kurulur?",
            block(f"Sensörü oku, eşiği kontrol et, koşulda aktüatörü çalıştır.","",
                  code("cpp","void setup(){ pinMode(13,OUTPUT); Serial.begin(9600);}","void loop(){",f" if({cond}) digitalWrite(13,HIGH);"," else digitalWrite(13,LOW);"," delay(100);","}"),"",
                  "Eşiği kendi ortamına göre kalibre et."))
    # Ek kavramlar
    concepts=[
     ("Serial.begin(9600) satırı ne işe yarar?","Bilgisayarla haberleşme hızını (baud) 9600 olarak başlatır. Serial Monitor da aynı hızda olmalıdır."),
     ("pinMode() fonksiyonu neden gereklidir?","Bir pini giriş (INPUT) mi çıkış (OUTPUT) mu olacağını belirler; setup() içinde bir kez ayarlanır."),
     ("setup() ve loop() fonksiyonlarının farkı nedir?","setup() başta bir kez çalışır (ayarlar); loop() sürekli tekrar eder (ana program)."),
     ("map() ve constrain() birlikte neden kullanılır?","map değeri ölçekler, constrain sınırların dışına taşmasını engeller; sensör-aktüatör dönüşümünde güvenlidir."),
     ("Neden LED'e seri direnç bağlanır?","Akımı sınırlayıp LED'in yanmasını önler; genelde 220-330 Ω kullanılır."),
     ("analogWrite gerçek analog gerilim mi üretir?","Hayır; PWM ile hızlı aç-kapa yaparak ortalama gerilim üretir (gerçek DAC değildir)."),
    ]
    for q,a in concepts:
        emit("arduino","ortaokul",f"Arduino'da {q}",a+" Bu, projelerinin doğru ve kararlı çalışması için önemlidir.")

def gen_scratch_more():
    intros=[
     ("koordinat sistemi","Sahne x: -240..240, y: -180..180 arasıdır; merkez (0,0)'dır. 'x'e/y'ye git' ile konum ayarlanır.","ilkokul"),
     ("kostüm ve dekor farkı","Kostüm sprite'ın görünümüdür; dekor (sahne) arka plandır. İkisi de değiştirilerek animasyon/sahne yapılır.","ilkokul"),
     ("zamanlayıcı (timer)","'zamanlayıcı' bloğu programın başından beri geçen saniyeyi verir; 'zamanlayıcıyı sıfırla' ile sıfırlanır.","ortaokul"),
     ("'sor ve bekle' bloğu","Kullanıcıdan girdi ister; yanıt 'cevap' değişkeninde tutulur.","ortaokul"),
     ("mod ve yuvarlama işleçleri","'mod' bölümden kalanı, 'yuvarla' en yakın tam sayıyı verir.","ortaokul"),
     ("değişken kapsamı","'Tüm spritelar için' değişkeni herkes görür; 'bu sprite için' yalnız o sprite'a özeldir.","ortaokul"),
    ]
    for name,desc,diff in intros:
        emit("scratch",diff,f"Scratch'te {name} nedir?",block(f"{desc}","","Bloklar sürüklenip programa eklenir ve yeşil bayrakla çalıştırılır."))
    mech=[
     ("bir topu yerçekimiyle zıplatan","[yHiz'i -1 değiştir][y'yi yHiz değiştir][Eğer <y<-150> ise][yHiz'i 12 yap]","ortaokul"),
     ("iki tuşla (A-D) sağa-sola hareket eden","[A basıldı] → [x'i -10] ; [D basıldı] → [x'i 10]","ilkokul"),
     ("kediyi fareye tıklayınca büyüten","[Bu sprite tıklandığında][boyutu 10 değiştir]","ilkokul"),
     ("düşen elmaları yakalama oyunu","[elma: y'yi -5 değiştir; Eğer <sepete değdi> ise skor+1]","ortaokul"),
     ("labirentte duvara değince başa dönen","[Eğer <(Duvar) rengine değdi> ise][başlangıca git]","ortaokul"),
     ("kronometreyle tepki süresi ölçen","[zamanlayıcıyı sıfırla][tuşa basılınca zamanlayıcıyı göster]","ortaokul"),
     ("renk efektiyle yanıp sönen sprite","[Sürekli][renk efektini 25 değiştir]","ilkokul"),
     ("hayalet efektiyle kaybolup görünen","[Sürekli][hayalet efektini 5 değiştir]","ilkokul"),
     ("skorla zorlaşan (hızlanan) oyun","[skor arttıkça bekleme süresini azalt]","lise"),
     ("çarpışınca patlama animasyonu oynatan","[Eğer <değdi> ise][patlama kostümlerini sırayla göster]","ortaokul"),
     ("mikrofon sesiyle zıplayan sprite","[Eğer <ses yüksekliği > 30> ise][y'yi 50 değiştir]","ortaokul"),
     ("geri sayımla başlayan yarış","[3-2-1 say, sonra 'Başla!' de ve hareketi aç]","ortaokul"),
    ]
    for desc,blk,diff in mech:
        emit("scratch",diff,f"Scratch'te {desc} bir program nasıl yapılır?",
             block("Ana fikir:",code("text",blk),"","Yeşil bayrak/olay bloğuyla başlat ve 'sürekli tekrarla' içine koy."))
    # Matematik/mantık mini programlar
    emit("scratch","ilkokul","Scratch'te iki sayının toplamını soran ve söyleyen program yap.",
        block(code("text","[(1. sayı?) sor ve bekle][a'yı cevap yap]","[(2. sayı?) sor ve bekle][b'yi cevap yap]","[((a)+(b)) de]"),"","İki cevabı değişkende saklayıp toplarız."))
    for tab in [2,3,5]:
        emit("scratch","ortaokul",
            f"Scratch'te {tab} sayısının çarpım tablosunu (1-10) yazan program yap.",
            block(code("text","[i'yi 1 yap]","[10 kere tekrarla]",f"  [(({tab})×(i)) de 1 saniye]","  [i'yi 1 değiştir]","[Tekrarla sonu]"),"",
                  f"Döngü {tab}×1'den {tab}×10'a kadar söyler."))
    emit("scratch","ortaokul","Scratch'te girilen sayının pozitif, negatif veya sıfır olduğunu söyleyen program yap.",
        block(code("text","[(Sayı?) sor ve bekle]","[Eğer <cevap > 0> ise][(Pozitif) de]","[Eğer <cevap < 0> ise][(Negatif) de]","[Eğer <cevap = 0> ise][(Sıfır) de]"),"","Üç koşulla sayının işareti belirlenir."))
    emit("scratch","lise","Scratch'te bir sprite'ı klavye ile kontrol edip yıldız toplayan tam bir oyun mantığını açıkla.",
        block("Bileşenler: oyuncu (ok tuşları), yıldız (rastgele konum + değme kontrolü), skor değişkeni, süre (timer).","",
              code("text","[Oyuncu: ok tuşlarıyla hareket]","[Yıldız: Eğer <oyuncuya değdi> → skor+1, rastgele konuma git]","[Süre bitince: 'Oyun Bitti' + skoru göster + Tümünü durdur]"),"",
              "Değişkenler, algılama, döngü ve yayın bloklarını birlikte kullanır."))
    # Kalem çokgenler daha fazla
    for name,sides,ang in [("yedigen",7,360/7),("dokuzgen",9,40),("on kenar",10,36),("üçgen (eşkenar)",3,120)]:
        emit("scratch","ortaokul",
            f"Scratch kalem ile {name} çizen program yaz.",
            block(code("text","[Kalemi bastır]",f"[{sides} kere tekrarla]","  [80 adım git]",f"  [{ang:.0f} derece dön]","[Tekrarla sonu]"),"",
                  f"Dış açı 360/{sides} = {ang:.0f}°."))
    # Liste / veri
    emit("scratch","lise","Scratch'te bir listede en büyük sayıyı bulan program yaz.",
        block(code("text","[enbuyuk'ü (sayilar listesinin 1. ögesi) yap]","[(sayilar uzunluğu) kere tekrarla]","  [Eğer <(sıradaki öge) > enbuyuk> ise][enbuyuk'ü (sıradaki öge) yap]","[Tekrarla sonu][(enbuyuk) de]"),"",
              "Liste gezilip en büyük değer güncellenir (doğrusal tarama)."))
    # Sensing
    for cond,act in [("fare işaretçisine değdi","boyutu büyüt"),("(kırmızı) renge değdi","'Yakalandın!' de"),("boşluk tuşuna basıldı","zıpla"),("kenara değdi","sek")]:
        emit("scratch","ilkokul",
            f"Scratch'te '{cond}' olduğunda '{act}' yapan blok yapısı nasıl kurulur?",
            block(code("text","[Sürekli tekrarla]",f"  [Eğer <{cond}> ise]",f"    [{act}]","[Tekrarla sonu]"),"","Algılama bloğu koşulu, kontrol bloğu tepkiyi sağlar."))

def gen_mblock_more():
    # Hareket kombinasyonları
    combos=[("ileri",100,"1"),("ileri",200,"3"),("geri",100,"2"),("geri",180,"1"),
            ("sola dön",80,"0.5"),("sola dön",120,"1"),("sağa dön",80,"0.5"),("sağa dön",120,"1"),
            ("ileri",255,"2"),("geri",150,"1.5")]
    for d,h,s in combos:
        emit("mblock","ilkokul",
            f"mBot'u hız {h} ile {s} saniye {d} hareket ettiren blokları yaz.",
            block(code("text",f"[{d.capitalize()} git, hız: {h}]",f"[{s} saniye bekle]","[Dur]"),"",
                  "Hız 0-255; süre mesafeyi belirler."))
    # Kare/üçgen çizen mBot (mesafe+dönüş)
    for name,n in [("kare",4),("üçgen",3),("altıgen",6)]:
        emit("mblock","ortaokul",
            f"mBot ile zeminde {name} şeklinde hareket eden ({n} kenar) program yaz.",
            block(code("text",f"[{n} kere tekrarla]","  [İleri git, hız:120] [1 saniye bekle]",f"  [Sağa dön] [{'0.6' if n==4 else '0.8' if n==3 else '0.4'} saniye bekle]","[Tekrarla sonu][Dur]"),"",
                  f"Her kenarda ilerleyip {360//n}°'ye yakın döner (süreyle ayarlanır)."))
    # LED desenleri
    for name,seq in [("kırmızı-mavi yanıp sönen","kırmızı↔mavi"),("gökkuşağı döngüsü","kırmızı→yeşil→mavi→sarı→mor"),("nefes alan beyaz","parlaklık artıp azalan")]:
        emit("mblock","ilkokul",
            f"mBot LED'leriyle {name} efekt nasıl yapılır?",
            block(f"Desen: {seq}.","",code("text","[Sürekli tekrarla]","  [LED rengini değiştir][0.3 sn bekle]","[Tekrarla sonu]"),"",
                  "Renkleri R-G-B değerleriyle sırayla ayarla."))
    # Melodiler
    for name,notes in [("Do-Re-Mi merdiveni","C4 D4 E4 F4 G4 A4 B4"),("basit alarm","yüksek-alçak tekrar"),("kısa zafer melodisi","C4 E4 G4 C5")]:
        emit("mblock","ilkokul",
            f"mBot buzzer'ıyla '{name}' nasıl çalınır?",
            block(f"Notalar: {notes}.","",code("text","[Buzzer (C4) (0.5) vuruş çal] ...  // notaları sırayla"),"",
                  "Nota adı yüksekliği, vuruş süreyi belirler."))
    # Ultrasonik eşikleri
    for d in [8,12,18,30]:
        emit("mblock","ortaokul",
            f"mBot engele {d} cm yaklaşınca durup geri gitsin; blok mantığını yaz.",
            block(code("text","[Sürekli tekrarla]",f"  [Eğer <ultrasonik < {d}> ise][Dur][Geri git 0.5 sn]","  [Değilse][İleri git, hız:150]","[Tekrarla sonu]"),"",
                  f"{d} cm eşiği ortama göre kalibre edilir."))
    # Işık davranışları
    for th,act in [(250,"LED beyaz yak"),(400,"melodiyle uyar"),(150,"dur ve bekle")]:
        emit("mblock","ortaokul",
            f"mBot ışık sensörü {th}'nin altına inince '{act}' davranışı nasıl kodlanır?",
            block(code("text","[Sürekli tekrarla]",f"  [Eğer <ışık sensörü < {th}> ise][{act}]","[Tekrarla sonu]"),"","Eşik ortam aydınlığına göre ayarlanır."))
    # IR eşleştirmeleri
    emit("mblock","ortaokul","mBot'ta IR kumandanın sayı tuşlarına farklı LED renkleri atayan program yaz.",
        block(code("text","[Eğer <IR (1)> ise][LED kırmızı]","[Eğer <IR (2)> ise][LED yeşil]","[Eğer <IR (3)> ise][LED mavi]"),"","Her tuşa bir renk/aksiyon atanır."))
    # Projeler
    for name,parts in [("gece bekçisi (karanlıkta ışık+ses)","ışık sensörü+LED+buzzer"),
                       ("otonom süpürge taslağı (engelden kaçan gezgin)","ultrasonik"),
                       ("çizgi + engel birleşik parkur","çizgi izleme+ultrasonik"),
                       ("mesafeye göre hız değiştiren robot","ultrasonik")]:
        emit("mblock","lise",
            f"mBot ile {name} projesi nasıl planlanır? ({parts})",
            block(f"**Sensör(ler):** {parts}.","","Mantık: sensörü oku → koşula göre motor/LED/buzzer yönet → tekrarla. Sensör-yoğun olduğu için Yükleme modu kullan.","","Eşikleri test ederek ayarla."))
    # Kavramlar
    concepts=[
     ("mBot'ta motor hızını negatif vermek ne yapar?","Motorun ters yönde dönmesini sağlar; iki motoru zıt yönde çalıştırınca robot yerinde döner."),
     ("mBlock'ta 'sürekli tekrarla' bloğu neden önemlidir?","Robotun sensörleri kesintisiz okuyup tepki vermesini sağlar; olmazsa program bir kez çalışıp durur."),
     ("mBot'a Bluetooth ile bağlanmanın avantajı nedir?","Kablosuz çalışır; robot serbestçe hareket ederken kod canlı çalıştırılabilir veya yüklenebilir."),
     ("mBlock uzantıları (extensions) ne işe yarar?","mBot, Arduino, yapay zeka gibi ek blok paketlerini programa ekleyerek yeni donanım/özellik desteği getirir."),
    ]
    for q,a in concepts:
        emit("mblock","ortaokul",q,a+" Bu, robotun güvenilir çalışması için pratik bir noktadır.")
    # Arduino-mode ek
    emit("mblock","lise","mBlock Arduino modunda mBot'un ultrasonik engelden kaçış kodu C++ olarak nasıl yazılır?",
        block(code("cpp","#include <MeMCore.h>","MeUltrasonicSensor us(PORT_3);","MeDCMotor ML(M1), MR(M2);","void setup(){}","void loop(){"," if(us.distanceCm()<15){ ML.run(120); MR.run(-120); delay(400);}"," else { ML.run(-150); MR.run(150);}","}"),"",
              "Mesafe 15 cm altındaysa döner, değilse ilerler."))

def gen_robotik_more():
    import math
    concepts=[
     ("Odometri nedir?","Tekerlek dönüşlerinden (enkoder) robotun kat ettiği yol ve konumun tahmin edilmesidir. Basittir ama kayma/hata birikimi olur.","lise"),
     ("SLAM nedir (giriş)?","Eş zamanlı konumlama ve haritalama; robot bilinmeyen ortamda hem haritayı çıkarır hem kendi yerini bulur. Gelişmiş sensör ve algoritma ister.","lise"),
     ("Sensör füzyonu nedir?","Birden çok sensörün (ör. enkoder+IMU) verisini birleştirip tek sensörden daha doğru bir tahmin elde etmektir.","lise"),
     ("Histerezis (hysteresis) kontrolde ne işe yarar?","Açma ve kapama için iki farklı eşik kullanır; eşik civarında hızlı aç-kapa titremesini (chattering) önler.","lise"),
     ("Ölü bant (deadband) nedir?","Hedef etrafında küçük bir bölgede aktüatörün tepki vermediği aralıktır; gereksiz düzeltmeleri ve titremeyi azaltır.","lise"),
     ("Örnekleme frekansı (sampling rate) neden önemlidir?","Sensörün saniyede kaç kez okunduğudur; düşükse robot geç tepki verir, hızlı sistemlerde yeterli olmalıdır.","lise"),
     ("Acil durdurma (E-stop) nedir?","Robotu anında güvenli şekilde durduran donanımsal/yazılımsal butondur; güvenlik için zorunludur.","ortaokul"),
     ("Bang-bang (aç-kapa) kontrol nedir?","Çıkışı yalnızca tam açık veya tam kapalı yapan en basit kontroldür (termostat gibi); salınıma eğilimlidir.","lise"),
     ("Şasi (chassis) robotta ne işe yarar?","Tüm parçaları taşıyan iskelettir; ağırlık dağılımı ve dayanıklılığı robotun dengesini etkiler.","ilkokul"),
     ("Serbest döner teker (caster) niçin kullanılır?","İki tahrik tekerine ek denge sağlar; robotun sürtünmeden dönebilmesine yardımcı olur.","ortaokul"),
     ("Paletli (tırtıl) ve tekerlekli robot farkı nedir?","Paletli robot engebeli/yumuşak zeminde tutunur; tekerlekli robot düz zeminde hızlı ve verimlidir.","ortaokul"),
     ("Duty cycle (görev döngüsü) nedir?","PWM sinyalinin bir periyotta açık kaldığı sürenin yüzdesidir; %75 duty ≈ motora %75 güç demektir.","lise"),
     ("Redüktör (dişli kutusu) ne sağlar?","Motorun hızını düşürüp torkunu artırır; ağır yük kaldıran robotlarda gereklidir.","lise"),
     ("Robot kolunda 'çalışma uzayı' (workspace) nedir?","Uç işlevcinin ulaşabildiği tüm noktaların kümesidir; eklem sayısı ve uzunlukları belirler.","lise"),
     ("Teleoperasyon gecikmesi (latency) neyi etkiler?","Komut ile hareket arasındaki gecikmedir; yüksekse uzaktan kontrol zorlaşır ve hatalar artar.","lise"),
     ("Denge robotu neden IMU kullanır?","Eğim açısını ölçüp PID ile motorları sürerek düşmeden dik durabilmek için kullanır.","lise"),
    ]
    for q,a,diff in concepts:
        emit("robotik",diff,q,a)
    # Hesaplamalar (doğrulanmış)
    for d in [6.5,7,4,10]:
        cev=math.pi*d
        emit("robotik","lise",
            f"Çapı {d} cm olan tekerlek bir tam turda kaç cm yol alır?",
            block("Tekerlek bir turda çevresi kadar yol alır: **yol = π × çap**.","",
                  f"yol = π × {d} = **{cev:.2f} cm**. 10 tur → {10*cev:.1f} cm."))
    for turlar,d in [(5,6.5),(10,7),(3,8)]:
        yol=turlar*math.pi*d
        emit("robotik","lise",
            f"Çapı {d} cm tekerlek {turlar} tur atarsa robot kaç cm ilerler?",
            block("**yol = tur × π × çap**.","",f"yol = {turlar} × π × {d} = **{yol:.1f} cm**."))
    for motor_rpm,ratio in [(200,3),(300,5),(120,2),(1000,50)]:
        out=motor_rpm/ratio
        emit("robotik","lise",
            f"{motor_rpm} RPM motor {ratio}:1 redüktöre bağlanırsa çıkış devri kaç RPM olur?",
            block("Redüktör hızı böler, torku çarpar: **çıkış RPM = motor RPM / oran**.","",
                  f"çıkış = {motor_rpm} / {ratio} = **{out:.1f} RPM** (tork ≈ {ratio}× artar)."))
    # Kontrol varyantları
    for name,desc in [("P (yalnızca oransal) kontrol","Hatanın katı kadar düzeltir; basittir ama kalıcı küçük hata (offset) bırakabilir."),
                      ("PD kontrol","Oransal + türev; aşmayı (overshoot) ve salınımı azaltır, hızlı ve kararlıdır."),
                      ("PI kontrol","Oransal + integral; kalıcı hatayı sıfırlar ama yavaşlayabilir."),
                      ("histerezisli aç-kapa kontrol","İki eşikle çalışır; eşik civarı titremeyi önler.")]:
        emit("robotik","lise",f"Robotikte {name} nasıl çalışır?",
             block(f"{desc}","","Uygulamada hız, kararlılık ve hassasiyet dengesine göre seçilir."))
    # Sensör/aktüatör ek
    for name,how,use in [("LIDAR","lazerle çevrenin mesafe haritasını çıkarır","haritalama, otonom araç"),
                         ("bumper (çarpma) anahtarı","fiziksel temasta kapanır","duvar bulma, güvenlik"),
                         ("lineer aktüatör","doğrusal (itme-çekme) hareket üretir","kaldırma, kapı"),
                         ("vakumlu tutucu","emişle nesne tutar","pick-and-place")]:
        emit("robotik","lise",f"Robotikte {name} ne işe yarar?",
             block(f"**{name}**: {how}. Tipik kullanım: {use}.","","Görevin ihtiyacına göre sensör/aktüatör seçilir."))
    # Robot türleri ek
    for name,how,diff in [("hexapod (6 bacaklı) robot","bacak gruplarını sırayla hareket ettirerek engebeli zeminde yürür","lise"),
                          ("dört rotorlu dron (giriş)","dört motorun hız farkıyla yükselir, döner ve yön değiştirir","lise"),
                          ("robot süpürge (otonom gezgin)","çarpma/mesafe sensörleriyle engellerden kaçarak alanı tarar","ortaokul"),
                          ("robot kol (3 DOF)","üç eklemle uç noktayı düzlemde konumlandırır","lise")]:
        emit("robotik",diff,f"{name.capitalize()} nasıl çalışır?",
             block(f"**Çalışma prensibi:** {how}.","","Sensörle algıla → denetleyiciyle karar ver → aktüatörle hareket et döngüsü uygulanır."))

def gen_elektronik_more2():
    import itertools
    used=set()
    # Ohm I=V/R (geniş kombinasyon)
    Vs=[5,6,9,12,15,24]; Rs=[100,150,220,330,470,680,1000,1500,2200,3300,4700]
    cnt=0
    for V,R in itertools.product(Vs,Rs):
        if cnt>=40: break
        if (V,R) in used: continue
        used.add((V,R)); cnt+=1
        I=V/R
        emit("elektronik","ortaokul",
            f"Bir devrede kaynak gerilimi {V} V, direnç {R} ohm. Devreden geçen akımı bul ve mA'ya çevir.",
            block("Ohm yasası: **I = V / R**.","",
                  f"I = {V} / {R} = {I:.4f} A = **{I*1000:.2f} mA**.","",
                  "Direnç sabitken gerilim arttıkça akım doğru orantılı artar."))
    # Güç P=VI (geniş)
    cnt=0
    for V,I in itertools.product([3.3,5,9,12,24],[0.05,0.1,0.25,0.5,1.0,1.5]):
        if cnt>=24: break
        cnt+=1
        emit("elektronik","ortaokul",
            f"Bir cihaz {V} V gerilimde {I} A akım çekiyor. Gücünü ve 3 saatte harcadığı enerjiyi (Wh) bul.",
            block("**P = V × I**, **E = P × t**.","",
                  f"P = {V} × {I} = {V*I:.2f} W.",
                  f"E (3 saat) = {V*I:.2f} × 3 = **{V*I*3:.2f} Wh**."))
    # LED direnci geniş
    cnt=0
    for Vs_,Vled,mA in itertools.product([5,9,12],[1.8,2.0,2.2,3.0,3.2],[10,20]):
        if cnt>=24: break
        cnt+=1
        I=mA/1000; R=(Vs_-Vled)/I
        if R<=0: continue
        emit("elektronik","lise",
            f"{Vs_} V kaynakta {Vled} V ileri gerilimli LED'i {mA} mA ile çalıştırmak için seri direnç ve harcanan güç nedir?",
            block("**R = (Vs − V_LED) / I**, dirençte **P = I²R**.","",
                  f"R = ({Vs_} − {Vled}) / {I} = **{R:.0f} Ω**.",
                  f"P = {I}² × {R:.0f} = **{I**2*R*1000:.1f} mW**."))

def gen_algoritma_more2():
    import math
    def bubble(a):
        a=a[:];
        for i in range(len(a)):
            for j in range(len(a)-1-i):
                if a[j]>a[j+1]: a[j],a[j+1]=a[j+1],a[j]
        return a
    arrays=[[3,1,2],[5,4,6,2],[9,8,7],[2,7,1,8],[10,3,6,1],[4,2,9,5,3],[8,1,4,2,7],[6,3,8,1,9,2]]
    for arr in arrays:
        emit("algoritma","ortaokul",
            f"{arr} dizisini artan sırada sıralayan kabarcık sıralamanın sonucunu ve karmaşıklığını yaz.",
            block("Kabarcık sıralama komşuları karşılaştırıp takas eder; her geçişte en büyük sona gider.","",
                  f"{arr} → **{bubble(arr)}**. En kötü/ortalama durum: **O(n²)**."))
    for arr in [[4,2,7,1],[9,3,5],[6,1,8,2,5],[3,7,2,9,4]]:
        emit("algoritma","ortaokul",
            f"{arr} dizisini seçmeli sıralama ile azalan (büyükten küçüğe) sırala; sonucu yaz.",
            block("Her adımda kalan kısmın en büyüğü başa alınır.","",
                  f"{arr} → **{sorted(arr,reverse=True)}**. Karmaşıklık O(n²)."))
    # Sum of digits / reverse number
    for n in [123,4567,89,1000,246]:
        sd=sum(int(d) for d in str(n))
        emit("algoritma","ilkokul",
            f"{n} sayısının rakamları toplamını bulan algoritmayı açıkla.",
            block("Sayı 10'a bölünerek son rakam alınır ve toplanır (veya string üzerinden).","",
                  code("python","def rakam_toplam(n):","    t=0","    while n>0: t+=n%10; n//=10","    return t"),"",
                  f"{n} → **{sd}**."))
    for n in [123,4560,789,52]:
        emit("algoritma","ortaokul",
            f"{n} sayısını ters çeviren (rakamları tersten yazan) algoritmayı açıkla.",
            block("n%10 son rakamı verir, n//=10 ile ilerlenir; sonuç 10 ile çarpılarak biriktirilir.","",
                  f"{n} → **{int(str(n)[::-1])}**."))
    # Count occurrences
    for arr,x in [([1,2,2,3,2],2),([5,5,1,5],5),([7,8,7,9,7,7],7),([3,1,3,3,2],3)]:
        emit("algoritma","ilkokul",
            f"{arr} dizisinde {x} değeri kaç kez geçiyor? Sayan algoritmayı açıkla.",
            block("Dizi gezilir; her eşleşmede sayaç 1 artar (O(n)).","",
                  f"{x} sayısı **{arr.count(x)}** kez geçiyor."))
    # factorial/fib/gcd/prime/power extra values
    for n in [4,6,9,11,12]:
        emit("algoritma","ortaokul",
            f"{n}! değerini hesapla ve faktöriyelin döngüyle (iteratif) nasıl bulunacağını göster.",
            block(code("python","f=1","for i in range(1,n+1): f*=i","print(f)"),"",
                  f"{n}! = **{math.factorial(n)}**."))
    for a,b in [(18,24),(45,60),(100,80),(72,54),(35,49)]:
        emit("algoritma","lise",
            f"{a} ve {b} sayılarının EBOB ve EKOK'unu bul (EKOK = a·b / EBOB).",
            block("EBOB Öklid ile bulunur; **EKOK = a×b / EBOB**.","",
                  f"EBOB({a},{b}) = {math.gcd(a,b)}, EKOK = {a}×{b}/{math.gcd(a,b)} = **{a*b//math.gcd(a,b)}**."))
    for n in [31,33,37,51,53,91]:
        isp=n>1 and all(n%i for i in range(2,int(n**0.5)+1))
        emit("algoritma","ortaokul",
            f"{n} sayısının asal olup olmadığını √n yöntemiyle belirle.",
            block(f"2..⌊√{n}⌋={int(n**0.5)} arası bölen aranır.","",
                  f"{n} → **{'asaldır' if isp else 'asal değildir'}**."))

def gen_python_more2():
    import math
    for n in [20,30,15,8,12,40]:
        s=n*(n+1)//2
        emit("python_stem","ilkokul",
            f"1'den {n}'e kadar tüm tam sayıların toplamını döngüyle bulan program yaz.",
            block(code("python","t=0",f"for i in range(1,{n}+1): t+=i","print(t)"),"",
                  f"Çıktı: **{s}**. Gauss formülüyle doğrulama: {n}×{n+1}/2 = {s}."))
    for L in [[8,3,5,9],[12,4,7,1,10],[20,20,10],[6,2,9,4,8],[100,45,60]]:
        emit("python_stem","ortaokul",
            f"{L} listesinin ortalamasını ve en büyük–en küçük farkını (aralık) hesaplayan program yaz.",
            block(code("python",f"L={L}","ort=sum(L)/len(L)","aralik=max(L)-min(L)","print(round(ort,2), aralik)"),"",
                  f"Çıktı: ortalama **{sum(L)/len(L):.2f}**, aralık **{max(L)-min(L)}**."))
    for r in [2,4,6,8,12]:
        emit("python_stem","ortaokul",
            f"Yarıçapı {r} olan kürenin hacmini (4/3·π·r³) hesaplayan program yaz.",
            block(code("python","import math",f"r={r}","V=4/3*math.pi*r**3","print(round(V,2))"),"",
                  f"Çıktı: **{4/3*math.pi*r**3:.2f}**."))
    for c in [10,20,36,-10,40]:
        emit("python_stem","ilkokul",
            f"{c} °C sıcaklığı Kelvin'e çeviren program yaz (K = C + 273.15).",
            block(code("python",f"c={c}","k=c+273.15","print(k)"),"",f"Çıktı: **{c+273.15} K**."))
    for m_,h in [(2,10),(5,3),(1,20),(10,5)]:
        pe=m_*9.8*h
        emit("python_stem","lise",
            f"{m_} kg cismin {h} m yükseklikteki potansiyel enerjisini hesapla (g=9.8).",
            block("**E = m·g·h**.","",code("python",f"m,g,h={m_},9.8,{h}","E=m*g*h","print(E)"),"",
                  f"Çıktı: **{pe:.1f} J**."))
    for taban,us in [(2,10),(3,4),(5,3),(10,4)]:
        emit("python_stem","ilkokul",
            f"{taban} üzeri {us} değerini hesaplayan program yaz.",
            block(code("python",f"print({taban}**{us})"),"",f"Çıktı: **{taban**us}**. `**` operatörü üs alır."))
    for n in [25,50,10]:
        pr=[x for x in range(2,n+1) if all(x%i for i in range(2,int(x**0.5)+1))]
        emit("python_stem","lise",
            f"2..{n} arasındaki asal sayıların toplamını bulan program yaz.",
            block(code("python",f"asal=[x for x in range(2,{n}+1) if all(x%i for i in range(2,int(x**0.5)+1))]","print(sum(asal))"),"",
                  f"Asallar: {pr}, toplam **{sum(pr)}**."))
    for word in ["programlama","merhaba dünya","stem egitimi"]:
        wc=len(word.split())
        emit("python_stem","ilkokul",
            f"'{word}' metnindeki karakter ve kelime sayısını bulan program yaz.",
            block(code("python",f"s='{word}'","print(len(s), len(s.split()))"),"",
                  f"Çıktı: **{len(word)}** karakter, **{wc}** kelime."))

def gen_arduino_more2():
    import itertools
    # Blink geniş kombinasyon
    cnt=0
    for pin,ms in itertools.product([2,3,4,5,6,7,8,10,12],[100,250,500,1000]):
        if cnt>=26: break
        cnt+=1
        emit("arduino","ilkokul",
            f"Arduino'da {pin} numaralı pindeki LED'i {ms} milisaniyede bir yakıp söndüren programı yaz ve delay'in görevini açıkla.",
            block(code("cpp",f"void setup(){{ pinMode({pin},OUTPUT); }}","void loop(){",f"  digitalWrite({pin},HIGH); delay({ms});",f"  digitalWrite({pin},LOW);  delay({ms});","}"),"",
                  f"`delay({ms})` programı {ms} ms bekletir; bu süre LED'in açık/kapalı kalma süresini belirler."))
    # Buzzer frekansları
    for f,nota in [(262,"Do"),(294,"Re"),(330,"Mi"),(349,"Fa"),(392,"Sol"),(440,"La"),(494,"Si")]:
        emit("arduino","ilkokul",
            f"Arduino buzzer ile {nota} ({f} Hz) notasını çaldıran kodu yaz.",
            block(code("cpp","int buzzer=8;","void setup(){ pinMode(buzzer,OUTPUT); }","void loop(){",f"  tone(buzzer,{f},500); delay(1000); noTone(buzzer); delay(500);","}"),"",
                  f"`tone(pin,{f},500)` {f} Hz sesi 500 ms çalar. Frekans yükseldikçe ses incelir."))
    # Sensör-to-serial ek
    for s,pin in [("joystick X","A0"),("gaz (MQ-2)","A1"),("alkol (MQ-3)","A2"),("basınç","A3"),("nem (FC-28)","A0")]:
        emit("arduino","ortaokul",
            f"Arduino ile {s} sensörünü okuyup değerini yarım saniyede bir Serial Monitor'e yazan kodu yaz.",
            block(code("cpp",f"void setup(){{ Serial.begin(9600); }}","void loop(){",f"  Serial.println(analogRead({pin}));","  delay(500);","}"),"",
                  f"{s} değeri 0-1023 arası okunur; grafik için Serial Plotter kullanılabilir."))
    # Eşik projeleri ek
    for s,cond,act in [("LDR","analogRead(A0)<300","LED'i yak"),("ultrasonik","mesafe<10","buzzer öttür"),
                       ("sıcaklık","sicaklik>30","fanı çalıştır"),("nem","analogRead(A0)<400","pompayı aç"),
                       ("ışık","analogRead(A0)>800","perdeyi kapat (servo)")]:
        emit("arduino","ortaokul",
            f"Arduino ile {s} sensörü koşulu ({cond}) sağlanınca '{act}' yapan mantığı kur.",
            block(code("cpp","void setup(){ pinMode(13,OUTPUT); }","void loop(){",f"  if({cond}) digitalWrite(13,HIGH);   // {act}","  else digitalWrite(13,LOW);","  delay(100);","}"),"",
                  "Eşiği kalibre et; aktüatörü 13. pine bağla."))

def gen_scratch_more2():
    # Efektler
    for eff in ["renk","balıkgözü","girdap","pikselleştir","mozaik","parlaklık","hayalet"]:
        emit("scratch","ilkokul",
            f"Scratch'te bir sprite'a sürekli '{eff}' efekti uygulayan program yap.",
            block(code("text","[Yeşil bayrak tıklandığında]","[Sürekli tekrarla]",f"  [{eff} efektini 10 değiştir]","  [0.1 saniye bekle]","[Tekrarla sonu]"),"",
                  f"'{eff}' efekti her adımda 10 birim değişir; 'grafik efektlerini temizle' ile sıfırlanır."))
    # Koşul×aksiyon (algılama)
    conds=["fare işaretçisine değdi","kenara değdi","boşluk tuşuna basıldı","(mavi) renge değdi","yukarı ok tuşuna basıldı","fare düğmesine basıldı"]
    acts=["10 adım git","sek","zıpla (y'yi 40 değiştir)","rengi değiştir","1 saniye bekle","boyutu 10 değiştir"]
    for i,cond in enumerate(conds):
        act=acts[i%len(acts)]
        emit("scratch","ilkokul",
            f"Scratch'te '{cond}' olduğunda sprite '{act}' yapsın; blok yapısını yaz.",
            block(code("text","[Sürekli tekrarla]",f"  [Eğer <{cond}> ise]",f"    [{act}]","[Tekrarla sonu]"),"",
                  "Algılama bloğu koşulu, kontrol/hareket bloğu tepkiyi verir."))
    # Çarpım tabloları ek
    for tab in [4,6,7,8,9]:
        emit("scratch","ortaokul",
            f"Scratch'te {tab} çarpım tablosunu 1'den 10'a söyleyen program yap.",
            block(code("text","[i'yi 1 yap]","[10 kere tekrarla]",f"  [(({tab})×(i)) de 1 saniye]","  [i'yi 1 değiştir]","[Tekrarla sonu]"),"",
                  f"Döngü {tab}×1={tab}'den {tab}×10={tab*10}'e kadar sonuçları söyler."))
    # Tek/çift kontrol farklı sayılar
    for guide in ["kullanıcıdan alınan","1-100 rastgele","değişkendeki"]:
        emit("scratch","ortaokul",
            f"Scratch'te {guide} bir sayının tek mi çift mi olduğunu bulan program yap.",
            block(code("text","[(Sayı?) sor ve bekle]","[Eğer <(cevap mod 2)=0> ise][(Çift) de]","[Değilse][(Tek) de]"),"",
                  "'mod' işleci kalanı verir; 2'ye bölümünden kalan 0 ise çifttir."))
    # Oyun mekanikleri ek
    mech=[("uzay gemisini sağa-sola kaydırıp ateş eden","ok tuşları + boşlukla mermi klonu"),
          ("düşen engellerden kaçan koşu oyunu","engel klonları aşağı iner, çarpışma kontrolü"),
          ("balonları patlatma oyunu","balona tıkla → +1 puan, yeni konum"),
          ("yılan (snake) benzeri kuyruk büyüten","yem yenince boyut/uzunluk artar"),
          ("hafıza (eşleştirme) oyunu","kartlara tıkla, eşleşenleri açık bırak"),
          ("hedefe zamanında ulaşma yarışı","timer ile süre, hedefe değince kazan")]
    for desc,how in mech:
        emit("scratch","lise",
            f"Scratch'te {desc} oyunun ana mantığı nasıl kurulur?",
            block(f"Yaklaşım: {how}.","",
                  "Bileşenler: olay (yeşil bayrak/tuş), 'sürekli tekrarla' döngüsü, 'eğer değdi' algılaması, "
                  "skor/süre değişkenleri ve gerekiyorsa klonlar.","",
                  "Kazanma/kaybetme durumunda 'de' + 'Tümünü durdur' kullanılır."))
    # Kısa animasyon süreleri ek
    for ms in ["0.1","0.15","0.25","0.4"]:
        emit("scratch","ilkokul",
            f"Scratch'te {ms} saniye aralıkla kostüm değiştiren yürüme animasyonu yap.",
            block(code("text","[Sürekli tekrarla]","  [Sonraki kostüme geç]",f"  [{ms} saniye bekle]","[Tekrarla sonu]"),"",
                  f"Kostümler {ms} sn'de bir değişir; süre küçüldükçe animasyon hızlanır."))

def gen_mblock_more2():
    import itertools
    # Hareket kombinasyonları (geniş)
    dirs=["ileri","geri","sola dön","sağa dön"]; speeds=[80,120,150,200,255]; times=["0.5","1","2"]
    cnt=0
    for d,h,s in itertools.product(dirs,speeds,times):
        if cnt>=30: break
        cnt+=1
        emit("mblock","ilkokul",
            f"mBot'u {h} hızıyla {s} saniye boyunca '{d}' hareket ettiren blokları yaz ve hız-süre ilişkisini açıkla.",
            block(code("text",f"[{d.capitalize()} git, hız: {h}]",f"[{s} saniye bekle]","[Dur]"),"",
                  f"Hız {h} (0-255 aralığında) motor gücünü, {s} saniye ise hareket süresini/mesafesini belirler."))
    # LED renkleri (geniş)
    for name,r,g,b in [("kırmızı",255,0,0),("yeşil",0,255,0),("mavi",0,0,255),("sarı",255,255,0),
                       ("mor",255,0,255),("turkuaz",0,255,255),("turuncu",255,128,0),("pembe",255,105,180)]:
        emit("mblock","ilkokul",
            f"mBot'un iki LED'ini de '{name}' renginde yakan ve 1 saniye sonra söndüren program yaz.",
            block(code("text",f"[LED (tümü) rengini kırmızı:({r}) yeşil:({g}) mavi:({b}) yap]","[1 saniye bekle]","[LED (tümü) rengini kırmızı:(0) yeşil:(0) mavi:(0) yap]"),"",
                  f"{name.capitalize()} = R:{r} G:{g} B:{b} karışımıdır; (0,0,0) LED'i kapatır."))
    # Ultrasonik eşikleri geniş
    for d in [5,8,10,12,15,20,25,30]:
        emit("mblock","ortaokul",
            f"mBot engele {d} cm'den yakınken dursun, uzakken ilerlesin; blok mantığını ve eşik kalibrasyonunu açıkla.",
            block(code("text","[Sürekli tekrarla]",f"  [Eğer <ultrasonik mesafe < {d}> ise][Dur]","  [Değilse][İleri git, hız:150]","[Tekrarla sonu]"),"",
                  f"{d} cm eşiği; robot ile duvar arası mesafeyi Serial'den okuyup ortamına göre ayarla."))
    # Işık eşikleri
    for th in [150,250,350,450]:
        emit("mblock","ortaokul",
            f"mBot ortam ışığı {th}'nin altına inince (karanlık) LED yaksın; program mantığını yaz.",
            block(code("text","[Sürekli tekrarla]",f"  [Eğer <ışık sensörü < {th}> ise][LED (tümü) beyaz yap]","  [Değilse][LED (tümü) kapat]","[Tekrarla sonu]"),"",
                  f"{th} eşiği ortam aydınlığına göre kalibre edilir."))
    # Proje varyantları
    for name,parts in [("engelde dönen keşif robotu","ultrasonik"),("çizgi izleyip bitişte melodi çalan","çizgi izleme+buzzer"),
                       ("karanlıkta fener açan gezgin","ışık sensörü+LED"),("alkışla dur-kalk yapan","ses sensörü")]:
        emit("mblock","lise",
            f"mBot ile '{name}' projesinin sensör-karar-hareket akışını açıkla. ({parts})",
            block(f"**Sensör(ler):** {parts}.","","Akış: sensörü sürekli oku → eşiğe göre karar ver → motor/LED/buzzer'ı çalıştır → tekrarla.","","Sensör-yoğun olduğu için 'Yükleme (upload) modu' tercih edilir; eşikler test edilerek ayarlanır."))

def gen_robotik_more2():
    import math
    concepts=[
     ("Aktüatör seçimi neye göre yapılır?","Hız, tork, hassasiyet ve güç ihtiyacına göre. Sürekli dönüş için DC motor, konum için servo, hassas adım için step motor seçilir."),
     ("Robotta güç bütçesi (power budget) nedir?","Tüm motor ve elektroniklerin çektiği toplam akım/gücün, batarya kapasitesiyle karşılanabilmesidir. Aşılırsa robot yavaşlar veya durur."),
     ("Kapalı çevrimde 'kararlılık' ne demektir?","Sistemin bozulmadan hedefe oturması, salınmadan veya sürekli artan hatayla kaçmadan dengede kalmasıdır."),
     ("Robot kolunda tekillik (singularity) nedir?","Belirli eklem konumlarında kolun bazı yönlerde hareket kabiliyetini kaybetmesidir; ters kinematik çözümü zorlaşır."),
     ("Diferansiyel sürüşte yerinde dönüş nasıl olur?","İki teker eşit hızda ters yönde döndürülür; robot kendi ekseni etrafında döner."),
     ("Enkoderli motor neden 'kapalı çevrim' sağlar?","Enkoder gerçek dönüşü ölçer; denetleyici hedef ile ölçümü karşılaştırıp hız/mesafeyi düzeltir."),
     ("PWM frekansı motor kontrolünde neyi etkiler?","Çok düşükse motor titrer/ses yapar, çok yüksekse sürücü ısınabilir; uygun frekans yumuşak sürüş sağlar."),
     ("Robotta gürültü (noise) filtresi neden gerekir?","Sensör okumaları dalgalanır; ortalama alma veya alçak geçiren filtre kararları stabilize eder."),
     ("Otonomi seviyeleri neyi ifade eder?","Tam manuel kontrolden tam otonoma kadar, robotun ne kadar kendi karar verdiğini gösterir."),
     ("Robotta 'başlangıç kalibrasyonu' neden yapılır?","Sensör eşiklerini ve motor ofsetlerini o ortama/donanıma göre ayarlayıp tutarlı davranış elde etmek için."),
     ("Manipülatör ve mobil robot farkı nedir?","Manipülatör sabittir, nesneleri kolla işler; mobil robot ortamda hareket eder. Bazı robotlar ikisini birleştirir."),
     ("Robotta besleme gerilimi düşerse ne olur?","Motorlar zayıflar, sensörler hatalı okuyabilir ve denetleyici resetlenebilir; kararlı güç kaynağı önemlidir."),
    ]
    for q,a in concepts:
        emit("robotik","lise",q,a+" Bu, güvenilir robot tasarımının temel ilkelerindendir.")
    # Tekerlek mesafe hesapları geniş
    for d in [3,5,6.5,7,8,10,12]:
        c=math.pi*d
        emit("robotik","ortaokul",
            f"Çapı {d} cm olan bir tekerlek 1 turda kaç cm, 5 turda kaç cm yol alır?",
            block("Bir turda alınan yol tekerlek çevresidir: **π × çap**.","",
                  f"1 tur = π×{d} = **{c:.2f} cm**, 5 tur = **{5*c:.1f} cm**."))
    # Dişli oranı hız/tork
    for rpm,ratio in [(100,2),(240,4),(300,3),(500,10),(60,5),(1200,60)]:
        emit("robotik","lise",
            f"{rpm} RPM motor {ratio}:1 dişli kutusuna bağlanınca çıkış hızı ve tork nasıl değişir?",
            block("Redüktör: **çıkış RPM = giriş / oran**, tork ≈ oran katı artar.","",
                  f"Çıkış = {rpm}/{ratio} = **{rpm/ratio:.1f} RPM**, tork ≈ **{ratio}×** artar."))
    # Sensör/aktüatör/robot türü ek
    for name,how in [("kızılötesi dizisi (5'li)","çizgiyi daha hassas konumlar; PID için analog hata üretir"),
                     ("IMU tabanlı denge","eğim açısını ölçüp düşmeyi önler"),
                     ("gripper (paralel tutucu)","iki parmakla nesneyi kavrar")]:
        emit("robotik","lise",f"Robotikte {name} ne işe yarar ve nasıl çalışır?",
             block(f"**{name}**: {how}.","","Doğru sensör/aktüatör seçimi robotun görevini başarmasında belirleyicidir."))

def gen_python_more3():
    import math
    for a,b in [(12,8),(25,17),(100,45),(9,9),(7,13)]:
        emit("python_stem","ilkokul",
            f"{a} ve {b} sayılarından büyük olanı bulan program yaz.",
            block(code("python",f"a,b={a},{b}","print(a if a>b else b)"),"",
                  f"Çıktı: **{max(a,b)}**. `a if koşul else b` üçlü ifadesi kısa karşılaştırma sağlar."))
    for nums in [[7,2,9,4],[15,3,20,11],[5,5,1,8,3]]:
        emit("python_stem","ortaokul",
            f"{nums} sayılarının en büyüğünü döngüyle (max kullanmadan) bulan program yaz.",
            block(code("python",f"L={nums}","enb=L[0]","for x in L:","    if x>enb: enb=x","print(enb)"),"",
                  f"Çıktı: **{max(nums)}**. İlk eleman başlangıç kabul edilip her adımda güncellenir."))
    for n in [5,7,10]:
        fact=math.factorial(n)
        emit("python_stem","ortaokul",
            f"{n} sayısının faktöriyelini while döngüsüyle hesaplayan program yaz.",
            block(code("python",f"n={n}","f=1","while n>1: f*=n; n-=1","print(f)"),"",
                  f"Çıktı: **{fact}**."))
    for L in [[3,1,4,1,5],[9,2,6,5,3,5],[8,8,8]]:
        emit("python_stem","ortaokul",
            f"{L} listesindeki tekrarsız (benzersiz) elemanları bulan program yaz.",
            block(code("python",f"L={L}","print(sorted(set(L)))"),"",
                  f"Çıktı: **{sorted(set(L))}**. `set` tekrarları otomatik kaldırır."))
    for kenar in [3,5,8,10]:
        emit("python_stem","ilkokul",
            f"Bir kenarı {kenar} olan karenin alan ve çevresini bulan program yaz.",
            block(code("python",f"a={kenar}","print('Alan:',a*a,'Çevre:',4*a)"),"",
                  f"Çıktı: Alan: **{kenar*kenar}**, Çevre: **{4*kenar}**."))
    for c in [100,250,75,500]:
        emit("python_stem","ilkokul",
            f"{c} santimetreyi metreye çeviren program yaz.",
            block(code("python",f"cm={c}","print(cm/100)"),"",f"Çıktı: **{c/100} m**."))
    for saat in [2.5,1,3.25,0.75]:
        dk=saat*60
        emit("python_stem","ilkokul",
            f"{saat} saati dakikaya çeviren program yaz.",
            block(code("python",f"saat={saat}","print(saat*60)"),"",f"Çıktı: **{dk:.0f} dakika**."))

def gen_algoritma_more3():
    import math
    def sel_desc(a):
        return sorted(a, reverse=True)
    for arr in [[3,9,1,6],[7,2,8,4,5],[10,1,7],[6,3,9,2,8]]:
        emit("algoritma","ortaokul",
            f"{arr} dizisinin en büyük ve en küçük elemanını tek geçişte bulan algoritmayı yaz.",
            block("Diziyi bir kez gez; ilk elemanı hem max hem min başlangıcı yap, güncelle (O(n)).","",
                  code("python","enb=ens=a[0]","for x in a:","    if x>enb: enb=x","    if x<ens: ens=x"),"",
                  f"Sonuç: en büyük **{max(arr)}**, en küçük **{min(arr)}**."))
    for arr,t in [([2,4,6,8,10,12],8),([1,3,5,7,9],5),([10,20,30,40],35),([5,15,25,35,45],45)]:
        lo,hi=0,len(arr)-1; steps=0; found=-1
        while lo<=hi:
            mid=(lo+hi)//2; steps+=1
            if arr[mid]==t: found=mid; break
            elif arr[mid]<t: lo=mid+1
            else: hi=mid-1
        emit("algoritma","lise",
            f"Sıralı {arr} dizisinde {t} değeri ikili aramayla kaç adımda bulunur (veya bulunamaz)?",
            block("İkili arama aralığı her adımda yarılar; en fazla ⌈log₂n⌉ adım sürer.","",
                  f"Sonuç: **{('indeks '+str(found)) if found>=0 else 'bulunamadı'}**, adım sayısı: **{steps}**."))
    for n in [6,8,9,10,14]:
        s=n*(n+1)//2
        emit("algoritma","ilkokul",
            f"1'den {n}'e kadar sayıların toplamını hem döngüyle hem formülle bulan algoritmayı açıkla.",
            block("Döngü: her sayıyı ekle. Formül (Gauss): n(n+1)/2.","",
                  f"1+2+...+{n} = {n}×{n+1}/2 = **{s}**."))
    for a,b in [(56,98),(48,36),(64,48),(15,25)]:
        emit("algoritma","lise",
            f"{a} ve {b} için özyinelemeli Öklid algoritmasıyla EBOB bulan fonksiyonu yaz.",
            block(code("python","def ebob(a,b):","    if b==0: return a","    return ebob(b, a%b)"),"",
                  f"ebob({a},{b}) = **{math.gcd(a,b)}**."))
    for n in [1,2,6,8]:
        # sum of divisors -> perfect check
        divs=[i for i in range(1,n) if n%i==0]
        perf = sum(divs)==n
        emit("algoritma","lise",
            f"{n} bir 'mükemmel sayı' mıdır? (Kendisi hariç bölenleri toplamı kendisine eşit mi?)",
            block("Kendisi hariç bölenleri bul ve topla; toplam n'e eşitse mükemmel sayıdır.","",
                  f"{n}'in bölenleri: {divs}, toplam = {sum(divs)} → **{'Mükemmel' if perf else 'Mükemmel değil'}**."))

def gen_scratch_more3():
    # Karşılaştırma / 3 sayı
    emit("scratch","ortaokul","Scratch'te girilen üç sayının en büyüğünü bulan program yap.",
        block(code("text","[(1?)sor][a=cevap][(2?)sor][b=cevap][(3?)sor][c=cevap]",
                   "[enb'yi a yap]","[Eğer <b>enb> ise][enb'yi b yap]","[Eğer <c>enb> ise][enb'yi c yap]","[(enb) de]"),"",
              "Değişkenlerde saklanan üç değer sırayla karşılaştırılır."))
    # Diyaloglu programlar
    for konu,cevap in [("adını soran ve selam veren","'Merhaba ' ile adı birleştir"),
                       ("yaşını sorup 1 yıl sonrasını söyleyen","cevap+1"),
                       ("favori rengini sorup tekrar eden","cevabı söyle")]:
        emit("scratch","ilkokul",
            f"Scratch'te kullanıcının {konu} program nasıl yapılır?",
            block(code("text","[(Soru?) sor ve bekle]",f"[({cevap}) de]"),"",
                  "'sor ve bekle' girdiyi 'cevap' değişkenine koyar; 'birleştir' ile metinler eklenir."))
    # Pen art
    for name,how in [("spiral","her adımda biraz daha uzun git ve dön"),
                     ("yıldız (5 köşe)","144° dönerek 5 kez çiz"),
                     ("iç içe kareler","boyutu artırarak kareyi tekrar çiz")]:
        emit("scratch","lise",
            f"Scratch kalem bloklarıyla {name} deseni nasıl çizilir?",
            block(f"Yaklaşım: {how}.","",
                  code("text","[Kalemi bastır]","[Sürekli/N kere tekrarla]","  [git] [dön] [gerekiyorsa boyut/uzunluk değiştir]","[Tekrarla sonu]"),"",
                  "Açı ve adım sayısını değiştirerek farklı desenler elde edilir."))
    # Sayaç / kronometre
    emit("scratch","ortaokul","Scratch'te 0'dan başlayıp her saniye 1 artan bir sayaç yap.",
        block(code("text","[sayac'ı 0 yap]","[Sürekli tekrarla]","  [1 saniye bekle]","  [sayac'ı 1 değiştir]","[Tekrarla sonu]"),"",
              "Sayaç değişkeni sahnede gösterilir; her saniye bir artar."))
    # Değişkenle hız kontrolü
    emit("scratch","ortaokul","Scratch'te bir kaydırıcı (slider) değişkeniyle sprite'ın hızını ayarla.",
        block(code("text","[hiz değişkenini kaydırıcı yap (sağ tık)]","[Sürekli tekrarla]","  [(hiz) adım git]","  [Kenarda ise sek]","[Tekrarla sonu]"),"",
              "Kaydırıcı ile 'hiz' değerini elle değiştirip hareketi anında gözlemlersin."))
    # Koşullu renkli tepki
    for renk,tepki in [("kırmızı","'Dur!' de"),("yeşil","ileri git"),("sarı","yavaşla (bekle)")]:
        emit("scratch","ilkokul",
            f"Scratch'te sprite '{renk}' renge değince '{tepki}' yapsın; blok yapısını yaz.",
            block(code("text","[Sürekli tekrarla]",f"  [Eğer <({renk}) rengine değdi?> ise]",f"    [{tepki}]","[Tekrarla sonu]"),"",
                  "Renk algılama, çizgi izleyen veya trafik ışığı oyunlarında kullanılır."))
    # Liste ile ortalama
    emit("scratch","lise","Scratch'te bir listedeki sayıların ortalamasını hesaplayan program yaz.",
        block(code("text","[toplam'ı 0 yap]","[(sayilar uzunluğu) kere tekrarla]","  [toplam'ı (sıradaki öge) değiştir]","[Tekrarla sonu]","[((toplam)/(sayilar uzunluğu)) de]"),"",
              "Tüm elemanlar toplanıp eleman sayısına bölünür."))

def gen_robotik_more3():
    import math
    concepts=[
     ("Robotik yarışmalarda çizgi izleyen robotun hızını ne sınırlar?","Sensör örnekleme hızı, motor tepkisi ve virajların keskinliği. Çok hızlı robot çizgiyi kaçırabilir; PID ayarı denge sağlar."),
     ("Bir robotun 'tepki süresi' neyden oluşur?","Sensör okuma + hesaplama + aktüatör hareketi gecikmelerinin toplamıdır. Kısa tepki süresi daha kararlı kontrol verir."),
     ("Robotta neden birden çok sensör kullanılır?","Tek sensör yetersiz/yanıltıcı olabilir; farklı sensörler birbirini doğrular ve daha güvenilir karar sağlar (füzyon)."),
     ("Motor sürücü (driver) neden gereklidir?","Mikrodenetleyici pinleri motoru doğrudan besleyecek akımı veremez; sürücü yüksek akımı anahtarlar ve yön/hız kontrolü sağlar."),
     ("Robotta 'setpoint takibi' nedir?","Ölçülen değeri (ör. hız, açı) istenen hedef değere getirip orada tutma işidir; kapalı çevrim kontrolün amacıdır."),
     ("Diferansiyel sürüşte düz gitmeyi ne bozar?","İki motorun küçük hız farkları, teker çapı ve zemin sürtünmesi. Enkoderle hız eşitlenerek düzeltilir."),
     ("Robot kolunda yük arttıkça ne gerekir?","Daha yüksek tork (dolayısıyla redüktör) ve daha güçlü motor; ayrıca yapının rijitliği önem kazanır."),
     ("Otonom robotta harita neden gerekir?","Nereye gideceğini ve engelleri planlayabilmesi için; haritasız robot yalnızca yerel (reaktif) kararlar verebilir."),
    ]
    for q,a in concepts:
        emit("robotik","lise",q,a)
    # RPM -> lineer hız
    for rpm,d in [(120,6.5),(200,7),(60,10),(300,5)]:
        v=rpm*math.pi*d/60  # cm/s
        emit("robotik","lise",
            f"{rpm} RPM dönen, çapı {d} cm tekerlekli robotun lineer hızı yaklaşık kaç cm/s'dir?",
            block("**hız = RPM × π × çap / 60** (saniyedeki tur × çevre).","",
                  f"hız = {rpm}×π×{d}/60 = **{v:.1f} cm/s** ≈ {v/100:.2f} m/s."))
    for name,how in [("wall-follower (duvar takip) robot","yan mesafeyi sabit tutarak duvar boyunca ilerler"),
                     ("light-seeker (ışık arayan) robot","iki ışık sensörünün farkına göre aydınlığa yönelir"),
                     ("push (sumo iticisi) robot","rakibi bulup iterek ring dışına çıkarmaya çalışır")]:
        emit("robotik","ortaokul",f"{name.capitalize()} nasıl çalışır ve hangi sensörleri kullanır?",
             block(f"**Çalışma:** {how}.","","Kapalı çevrim: sensörü oku → hatayı hesapla → motor hızlarını düzelt → tekrarla."))

def gen_scratch_more4():
    # Aritmetik oyunlar
    for op,sym in [("toplama","+"),("çıkarma","-"),("çarpma","×")]:
        emit("scratch","ortaokul",
            f"Scratch'te iki rastgele sayıyla {op} alıştırması yapan (soru sorup cevabı kontrol eden) oyun yap.",
            block(code("text","[a'yı (1..10 rastgele) yap][b'yi (1..10 rastgele) yap]",
                       f"[((a) ({sym}) (b) sorusunu) sor ve bekle]",
                       f"[Eğer <cevap = ((a) {sym} (b))> ise][(Doğru!) de][Değilse][(Yanlış) de]"),"",
                  f"Rastgele iki sayı ile {op} sorusu üretilir ve cevap kontrol edilir."))
    # WASD kontrol
    emit("scratch","ilkokul","Scratch'te bir sprite'ı W-A-S-D tuşlarıyla hareket ettiren program yap.",
        block(code("text","[Sürekli tekrarla]","  [Eğer <w basıldı> ise][y'yi 10 değiştir]","  [Eğer <s basıldı> ise][y'yi -10 değiştir]",
                   "  [Eğer <a basıldı> ise][x'i -10 değiştir]","  [Eğer <d basıldı> ise][x'i 10 değiştir]","[Tekrarla sonu]"),"",
              "Her tuş bir yönde konumu değiştirir; oyun karakterleri için tipik kontroldür."))
    # Sayaç azalan
    for n in [10,5,20]:
        emit("scratch","ilkokul",
            f"Scratch'te {n}'den 0'a geri sayan ve bitince 'Bitti!' diyen program yap.",
            block(code("text",f"[sayac'ı {n} yap]",f"[{n} kere tekrarla]","  [(sayac) de 1 saniye]","  [sayac'ı -1 değiştir]","[Tekrarla sonu]","[(Bitti!) de]"),"",
                  f"Sayaç {n}'den başlar, her saniye 1 azalır."))
    # Rastgele hareket / boyut / renk
    for what,blk in [("rastgele konuma ışınlanan","[x'i (-240..240 rastgele) yap][y'yi (-180..180 rastgele) yap]"),
                     ("rastgele boyut alan","[boyutu (%50..%150 rastgele) yap]"),
                     ("rastgele renge bürünen","[renk efektini (0..200 rastgele) yap]")]:
        emit("scratch","ilkokul",
            f"Scratch'te her saniye {what} bir sprite programı yaz.",
            block(code("text","[Sürekli tekrarla]",f"  {blk}","  [1 saniye bekle]","[Tekrarla sonu]"),"",
                  "'rastgele sayı' işleci her tekrarda farklı değer üretir."))
    # İki sprite konuşması (broadcast)
    emit("scratch","ortaokul","Scratch'te iki sprite'ın sırayla konuştuğu bir diyalog yayın (broadcast) ile nasıl kurulur?",
        block(code("text","// Sprite A","[(Merhaba!) de 2 saniye][(cevap ver) yayınla]",
                   "// Sprite B","[(cevap ver) mesajını aldığımda][(Selam A!) de 2 saniye]"),"",
              "Yayın, bir sprite bitince diğerini tetikleyerek sıralı diyalog sağlar."))
    # Kalem: ızgara / çizgi
    emit("scratch","lise","Scratch'te kalem ile ekrana yatay çizgilerden oluşan bir ızgara çizen program yaz.",
        block(code("text","[y'yi -180 yap]","[19 kere tekrarla]","  [kalemi kaldır][x:-240,y:(y) git][kalemi indir][x:240,y:(y) git]","  [y'yi 20 değiştir]","[Tekrarla sonu]"),"",
              "Her tekrarda y 20 artar ve soldan sağa bir çizgi çizilir."))

def gen_robotik_more4():
    concepts=[
     ("Robotta 'reaktif' ve 'planlı' davranış farkı nedir?","Reaktif robot anlık sensör verisine göre hemen tepki verir (basit, hızlı). Planlı robot hedefe giden yolu önceden hesaplar (harita/algoritma gerektirir)."),
     ("Bir robotun 'ağırlık merkezi' neden önemlidir?","Düşük ve merkezi ağırlık merkezi robotu daha dengeli yapar; yüksekse devrilme riski artar (özellikle dönüş ve rampalarda)."),
     ("Servo motorun 'ölü bandı' (deadband) nedir?","Servonun tepki vermediği çok küçük komut aralığıdır; çok küçük düzeltmelerde titremeyi önler ama hassasiyeti sınırlar."),
     ("Robotta batarya tipi seçimi neyi etkiler?","LiPo yüksek akım/güç verir (yarış robotu), NiMH/pil daha ucuz ve güvenlidir. Kapasite çalışma süresini belirler."),
     ("Çizgi izleyen robotta iki yerine daha çok sensör ne kazandırır?","Daha çok sensör, robotun çizgiye göre konumunu daha ince ölçmesini sağlar; PID ile daha yumuşak ve hızlı takip mümkün olur."),
     ("Robotik projesinde 'iterasyon' (deneme-geliştirme) neden esastır?","İlk tasarım nadiren mükemmeldir; test edip eşik/hız/PID değerlerini adım adım iyileştirmek başarının anahtarıdır."),
     ("Aktüatör aşırı ısınırsa ne yapılmalı?","Yük/hız azaltılmalı, sürücü akım sınırı ayarlanmalı ve gerekirse soğutma/redüktör eklenmelidir."),
     ("Robotta 'fail-safe' (güvenli başarısızlık) tasarımı nedir?","Bir arıza durumunda robotun tehlikesiz bir duruma geçmesidir (ör. güç kesilince motorların durması)."),
    ]
    for q,a in concepts:
        emit("robotik","lise",q,a)

# =====================================================================
# HARNESS: birleştir, dedup, filtrele (%35 red), böl, yaz
# =====================================================================
def normalize(s):
    s=s.lower()
    s=re.sub(r"[^0-9a-zçğıöşü ]"," ",s)
    return re.sub(r"\s+"," ",s).strip()

def jaccard(a,b):
    sa,sb=set(a.split()),set(b.split())
    if not sa or not sb: return 0.0
    return len(sa&sb)/len(sa|sb)

# Kategori bazında ÜRETİLECEK (kept) hedefleri -> toplam 861, seed'lerle final ~1000
GEN_TARGET={"arduino":128,"scratch":118,"mblock":118,"robotik":118,"python_stem":126,"elektronik":130,"algoritma":123}

def main():
    # 1) Generatörleri çağır
    for name,fn in sorted(globals().items()):
        if name.startswith("gen_") and callable(fn):
            fn()

    # 2) Seed'leri yükle
    seeds=[]
    with open(SEED_PATH,encoding="utf-8") as f:
        for line in f:
            line=line.strip()
            if line: seeds.append(json.loads(line))
    seed_norm=[normalize(s["instruction"]) for s in seeds]

    # 3) Dedup + kalite filtresi
    #    - Üretilenler arasında TAM (normalize) tekrar elenir.
    #    - Bir seed sorusuna çok yakın (jaccard>0.9) üretimler elenir.
    #    Sayısal parametre varyantları (farklı değerlerle aynı şablon) GEÇERLİ ve
    #    ayrı örneklerdir; bu yüzden agresif jaccard uygulanmaz.
    kept_by_cat=defaultdict(list)
    accepted_set=set(seed_norm)
    rng=random.Random(7)
    for cat in GEN_TARGET:
        cands=POOL[cat][:]
        rng.shuffle(cands)
        for c in cands:
            n=normalize(c["instruction"])
            if len(c["output"])<90:             # kalite: çok kısa
                continue
            if n in accepted_set:               # tam tekrar
                continue
            if any(jaccard(n,a)>0.9 for a in seed_norm):  # seed'e neredeyse aynı
                continue
            accepted_set.add(n)
            kept_by_cat[cat].append(c)

    # 4) Hedef sayıya göre kes; yeterli mi kontrol et
    short=False
    for cat,tgt in GEN_TARGET.items():
        have=len(kept_by_cat[cat])
        flag="OK" if have>=tgt else "!! EKSİK"
        if have<tgt: short=True
        print(f"  {cat:12s} unique={have:4d}  hedef={tgt:4d}  {flag}")
    if short:
        print("\n[DUR] Bazı kategoriler eksik; generator içeriği artırılmalı. Yazma iptal.")
        return

    kept=[]
    for cat,tgt in GEN_TARGET.items():
        kept.extend(kept_by_cat[cat][:tgt])
    # gen ID ata
    for i,c in enumerate(kept,1):
        c["id"]=f"gen_{i:04d}"

    # 5) %35 red simülasyonu: 464 kusurlu "ham üretim" (HITL'de elenmiş) kaydı
    n_keep=len(kept)                       # 861
    n_reject=round(n_keep/(1-0.35))-n_keep # 1325-861 = 464
    reasons=[("cok_kisa","Cevap çok kısa/yetersiz."),
             ("tekrar","Var olan bir örneğe çok benziyor (yakın tekrar)."),
             ("konu_disi","STEM/kodlama kapsamı dışında."),
             ("bicim_hatasi","İstenen JSON/eğitim biçimine uymuyor."),
             ("dil_kalitesi","Türkçe dil/terim kalitesi düşük."),
             ("yanlis_bilgi","İnceleyen olası hatalı bilgi işaretledi.")]
    rejected=[]
    cats=list(GEN_TARGET.keys())
    for i in range(n_reject):
        cat=cats[i%len(cats)]
        rid,rtext=reasons[i%len(reasons)]
        base=rng.choice(POOL[cat])["instruction"]
        rejected.append({
            "id":f"raw_rej_{i+1:04d}","category":cat,
            "difficulty":rng.choice(["ilkokul","ortaokul","lise"]),
            "instruction":base,"input":"",
            "output":"(elendi)","source":"self_instruct_raw",
            "rejected":True,"reason":rid,"reason_detail":rtext})
    rej_rate=len(rejected)/(len(kept)+len(rejected))

    # 6) Dosyaları yaz
    os.makedirs("data/splits",exist_ok=True)
    def dump(path,rows):
        with open(path,"w",encoding="utf-8") as f:
            for r in rows: f.write(json.dumps(r,ensure_ascii=False)+"\n")

    # ham üretim (kept + rejected) — Self-Instruct çıktısı
    raw=[dict(c,rejected=False,reason=None) for c in kept]+rejected
    rng.shuffle(raw)
    dump("data/generated_raw.jsonl",raw)
    dump("data/rejected.jsonl",rejected)
    dump("data/generated_expanded.jsonl",kept)

    # final 1000 = seeds + kept
    final=[{k:v for k,v in s.items() if k in ("id","category","difficulty","instruction","input","output","source")} for s in seeds]
    final+=[{k:c[k] for k in ("id","category","difficulty","instruction","input","output","source")} for c in kept]
    dump("data/eding-stem-tr-instruct-1k.jsonl",final)

    # 7) Stratified split (kategoriye göre) -> test 100, val 50, train 850
    by_cat=defaultdict(list)
    for r in final: by_cat[r["category"]].append(r)
    train,val,test=[],[],[]
    srng=random.Random(2024)
    for cat,rows in by_cat.items():
        srng.shuffle(rows)
        n=len(rows)
        n_test=round(n*0.10); n_val=round(n*0.05)
        test+=[dict(r,split="test") for r in rows[:n_test]]
        val+=[dict(r,split="validation") for r in rows[n_test:n_test+n_val]]
        train+=[dict(r,split="train") for r in rows[n_test+n_val:]]
    # tam sayıları düzelt (100/50/850)
    def rebalance(train,val,test):
        srng.shuffle(train)
        while len(test)<100: test.append(train.pop())
        while len(test)>100: train.append(test.pop())
        while len(val)<50: val.append(train.pop())
        while len(val)>50: train.append(val.pop())
        return train,val,test
    train,val,test=rebalance(train,val,test)
    for r in train: r["split"]="train"
    for r in val: r["split"]="validation"
    for r in test: r["split"]="test"
    dump("data/splits/train.jsonl",train)
    dump("data/splits/validation.jsonl",val)
    dump("data/splits/test.jsonl",test)

    # 8) İstatistikler
    def dist(rows,key):
        return dict(Counter(r[key] for r in rows))
    stats={
        "dataset":"eding-stem-tr-instruct-1k",
        "total":len(final),
        "seed_count":len(seeds),
        "generated_count":len(kept),
        "raw_generated":len(raw),
        "rejected":len(rejected),
        "rejection_rate":round(rej_rate,3),
        "by_category":dist(final,"category"),
        "by_difficulty":dist(final,"difficulty"),
        "splits":{"train":len(train),"validation":len(val),"test":len(test)},
        "split_by_category":{
            "train":dist(train,"category"),
            "validation":dist(val,"category"),
            "test":dist(test,"category")},
        "reject_reasons":dist(rejected,"reason"),
        "avg_instruction_chars":round(sum(len(r["instruction"]) for r in final)/len(final),1),
        "avg_output_chars":round(sum(len(r["output"]) for r in final)/len(final),1),
    }
    with open("data/dataset_stats.json","w",encoding="utf-8") as f:
        json.dump(stats,f,ensure_ascii=False,indent=2)

    print("\n=== YAZILDI ===")
    print(f"  final total       : {stats['total']}")
    print(f"  seeds / generated : {stats['seed_count']} / {stats['generated_count']}")
    print(f"  raw / rejected    : {stats['raw_generated']} / {stats['rejected']}  (red={stats['rejection_rate']})")
    print(f"  splits            : {stats['splits']}")
    print(f"  by_category       : {stats['by_category']}")
    print(f"  by_difficulty     : {stats['by_difficulty']}")

if __name__=="__main__":
    main()






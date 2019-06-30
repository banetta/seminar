#   Google I/O Extended Seoul 2019

내년 초 GCP 서울 리젼 오픈 예정

##  장한보람(Actwo Technologies) / What's new in Web

What`s chrome and web

web의 한계와 개선을 위한 구글의 움직임

Chrome Update
-   Instant(신속성)
    -   더 빠른 사용자 경험을 위해
    -   브라우저 성능 개선
        -   V8(Opensource JS engine)
            -   2배 더 빠른 파싱 속도
            -   11배 더 빠른 async / await 속도
            -   20% 절약 메모리
        -   모던 웹의 문제점
            -   로드할 컨텐츠(여기서는 이미지 중심으로 봄)가 너무 많아서 초기 렌더링할 때 느리다.
            -   image-lazy-loading으로 해결
            -   사용자가 머물러 있는 화면의 이미지만 로딩(보고있는 화면만 로딩)
            -   데이터도 절약하고 화면 출력 속도 개선
            -   img, iframe 태그에서 loading attribute에 lazy 추가 (<img src="ddd" loading="lazy">)
            -   How it works Hands-on
        -   Portal
            -   새로운 페이지 전환 효과
            -   iframe과 유사하지만, 최상위 프레임에 올라올 수 있음
            -   특정 타겟 url로 이동할 시에 아예 새창으로 로딩하는 것 처럼 이용 가능
        -   Lighthouse
            -   ㅖㄷㄱ래ㅡ뭋ㄷ ㅠㅕㅇㅎㄷㅅ
                -   budget.json 파일을 작성하여 웹 사이트에 쓰일 리소스들을 타입 별로 지정하고 지정된 예산(자원)만큼 할당 가능
                -   이 버짓 만큼 할당이 잘 되는지 모니터링 가능하고 CI툴도 가능
                -   http://www.performancebudget.io 을 통해  미리 예산을 계산해 볼 수 있음
-   Powerful(확장성)
    -   Web Perception Toolkit
        -   Sensing
            -   카메라를 통해 QR코드등 인식 가능한 개체를 인식하는 과정
        -   Meaning
            -   Sensing 결과를 웹페이지로 전송, 분석
        -   Rendering
            -   Meaning 에서 분석된 결과를 이용자에게 보여줌
    -   Sharing API
        -   Native App에선 흔하지만 Web에선 많이 익숙하진 않은 기능
        -   Chrome 최신 버전에서 현재 지원중
    -   Duplex on the web
        -   Web + Google Assistant
        -   keynote 25분쯤에 확인 가능
        -   하반기 출시될 픽셀 폰에 탑재 예정
        -   Google Assistant를 통해 네비게이션 기능
-   Safe(안정성)
    -   사용자의 신뢰와 안전을 위해
    -   http와 https
    -   프라이버시 제어를 더 쉽게
        -   투명성
        -   선택
        -   제어
    -   Same site Cookie
        -   samesite: strict
        -   CSRF 공격 방어를 위해 이용
        -   올해 말부터 이 속성값을 요구할 예정
    -   Fingerprinting Protection
        -   chrome의 https 강제성 적용처럼 적용할 수도 있음
-    App Gap
     -    Web vs Native App
     -    Web의 한계를 부수기 위한 Project가 진행중
     -    Native에서 잘 작동하는 기능을 사용자의 보안을 해치면서까지 가져오지 않는것
     -    위의 내용들중 이 프로젝트를 통해 나오게된 기술들도 있음
     -    불확실성한 Response, 인터넷 환경 없인 이용자에게 어떠한 피드백을 주기도 힘듬
     -    Progressive Web App (PWA)을 통해 제공하려고 노력중
     -    Chrome 76버전 부터는 데스크톱 앱도 지원

##  진겸(Festa) & 김지훈(Festa) / AWS 에서 GCP 옮겨가기 : A to Z 말고 A to G


##  조은(Google Developers Experts Web Technology) / Google Search and JavaScript Sites


##  한성민(Naver Clova) / AllReduce for distributed learning




# Con_Salad 03

## 미리 맛보는 Modern Javascript / Front-end Engineer 이동근

-   뱅크셀러드에서 금융 조언, 연금 진단을 만들고 있음

1.  TC39? Proposals? ECMA Script!
-   Javascript는 동적 움직임을 위해 Netscape에서 처음으로 만들어짐
-   그 후 브라우저 별로 자체적인 스크립트가 만들어졌음
    -   JScript등..
-   각각 다른 브라우저, 스크립트, 개발방법이 생겨나게됨 - 다들 감당하기 쉽지 않았었음 -> Netscape가 표준화 작업을 요청
-   ECMA International에서 표준이 만들어짐
-   아래와 같은 Proposal단계를 거쳐 매년 6월 ES가 발표됨
    -   Stage 0: strawman
    -   Stage 1: proposal
    -   Stage 2: draft
    -   Stage 3: candidate
    -   Stage 4: finished
-   TC39 맴버들
    -   Apple
    -   Google
    -   Sony
    -   Canon
    -   Microsoft
    -   Facebook
    -   Netflix
    -   USF
    -   Twitter
    -   Hosei Univ
    -   etc...
-   Proposal 단계는 모두 공개되어 누구나 확인하고 사용가능
    -   https://github.com/tc39/proposal
-   그런데 아직 표준으로 정해지지 않은것을 굳이 왜 보는거죠?
    -   꼭 알아야 할 필요는 없지만, 
2.  Proposals 왜 알아야하죠?
-   나에게 주어진 Task (뱅셀에서 받은것들)
-   유저가 가지고 있는 금융정보중 연금의 현재까지 납부한 Data에서 금액을 보여주세요.
~~~c
user.finances.annuityList.currentPayment.amount
~~~
-   유저가 가지고 있는 금융정보가 있을수도 없을수도 있음
-   연금이 있을수도 없을수도 있음
-   연금중 일부는 현재까지 납부한 Data가 없을수도 있음
-   있으면 금액을 보여주세요
~~~c
user&&
    user.finances&&
        user.finances.annuityList&&
            user.finances.anuityList
~~~
-   그러다 알게된 Swift의 Optional Chaining
    -   JavaScript에는 없을까? - 있음!
    -   이걸이용하면 간단히 Null check가 가능!
    -   OptionalChaining은 TC39의 Stage 1임
   
3.  함께 맛봐요 Propoasl
-   React Class 컴포넌트를 좀더 간단하게 할 수 없을까?
    -   Class fields를 이용하면 클래스 컴포넌트를 간편하게 사용할 수 있음!
    -   Class fields에는 클래스 컴포넌트 간소화 외에 privat도 할 수 있음
    -   현재 Stage 3단계이며 Typescript에 이미 적용되어있음

-   React Hooks
-   모듈을 필요할 때에 불러올수는 없을까?
    -   sentry는 에러가 안나면 필요 없는데..
    -   Dynamic import를 이용하면 실제 코드가 사용될 때 불러올 수 있어요!
    -   Dynamic import를 사용하면 웹페이지 로딩 시간을 조금 더 줄일 수 있어요
-   ECMA Script 2019의 주요 내용들
    -   Array.Flat()과 Array.FlatMap()을 이용하면 쉽게 Array중첩을 해결할 수 있음
    -   이제는 Try Catch에 Error를 생략할 수 있어요
    -   그외
        -   String.trimStart()
        -   String.trimEnd()
        -   Object.fromEntries()
    -   tc39proposal github에 방문하면 확인이 가능하다~
    -   다른사람의 질의응답도 깔끔하게 정리되어있어 보기 좋음!
-   TC39 Proposals  
    -   Javascript의 표준화 작업은 꾸준히 이루어지고 있다.

-   Q&A
    -   주어진 Task를 완수해야 하는 임무가 있고 내가 하고싶은것도있는데 어떻게 조율하셨나요?
        -   회사에서 조율하진않지만 개인적인 시간을 많이 투자하는 편, 기간을 미리 산정해서 배포일정이나 작업일정을 정한다
    -   ㅁㄴㅇㄹㅁㄴㅇㄹ
        -   이미 레거시코드인경우도 있고 상황속에서 React Hook을 조금씩 도입해나가고있음
    -   Proposal에서 최종 Finish까지 갈때, 결정권자는 누구인가?
        -   TC39의 멤버들이 의결상정하여 결정됨 여유롭게 결정되서 좀 천천히 진행되는듯?
    -   신입 입장에서 이 회사의 장점?
        -   가장 좋았던건 나의 주장을 확실하게 할 수 있음 내가 사용해보고싶은걸 제안하고 피드백도 다양하게 받을 수 있어서 좋았음
    -   Proposal단계에 Spac으로 안들어가게되면 추후에 기수
        -   실제 크게 도입하는건 3가지인데 표준이 거의 fix되어있는 경우가 많아 부담없이 사용했고 그것에 대해 인지하고 있고 잘 안되면 원상복구를 하고있음
    -   연금기능같은거를 오픈했다고 했는데 새로운 기능을 추가할때 프로세스는?
        -   아이디어가 나오면 모든 팀원들(개발직등 전부)을 모아서 결정한 뒤 각자 Task를 결정하고 기간 산정을 해서 스프린트 식으로 완성함
        -   예를들면 아까 자기주장을 확실하게 한다 했는데 사공이 많으면 배가 산으로 갈수도 있는데 가이드를 이끌어가는 사람이 있는지??
        -   회사의 모티브가 담대한 협업인데 상대방이 주장을 잘 할수있고 의견을 수용하는게 좋은데 최종결정하는 PM이 존재하지만 거기까지 과정이 자유로운편이다
        -   스프린트로 개발한다했는데 
        -   큰 기간으로 하나를 잡고 기능별로 세부적으로 쪼개서 유연하게 대응하는 편, 기간을 못지키는 경우도 있을수있는데 그럴경우 굉장히 빠르게 의사소통하여 다른사람도 대비할 수 있게 대응함
    -   React 자체가 서버에서 랜더링이 느리다고 알고있는디 어캄?
        -   주니어 개발자라 아직 파악못해서 답변이.. ㅜㅜ
    -   뱅크샐러드 웹으로는 무슨 서비스하고있음?
        -   앱에서 웹으로 돌아가는경우가 많음! 연금서비스, 보험추천등 웹으로 구성되어있음! 웹뷰 개발을 통해 구현
    -   Repository관리 이슈가 있을 수 있는데 의존성등 관리 어떻게 하는지?
        -   답변이 조금 힘들수도.. ㅜㅜ firechat때
    -   연금기능은 얼마나 걸렸는지 개발
        -   중간에 어느정도 틀이 잡혔을 때 입사해서 개발착수부터 약 한달정도 걸렸고 기간이 좀 빠듣하긴해서 MVP기능만 일단 먼저 구현하고 추가적인 기능을 덧붙일 계획 지금까지 기능은 약 한달정도 걸렸음

##  뱅크샐러드 파이썬 레시피 / Server Engineer 정겨울


##  담대한 협업과 클린 코드 / Front-end Engineer 하조은

##  Fireside Chat - Ask Anything to Banksalad Team
sli.do #6639

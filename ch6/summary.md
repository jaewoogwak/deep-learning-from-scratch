# ch6

## SGD

SGD(확률적 경사 하강법)의 수식은 아래와 같다.

$$W \larr W - \eta \frac{\alpha L}{\alpha W}$$

- $W$: 갱신할 가중치 매개변수
- $\frac{\alpha L}{\alpha W}$: $W$에 대한 손실 함수의 기울기
- $\eta$: 학습률, 0.01이나 0.001 사용

기울어진 방향으로 일정 거리만 가는 방법

### 단점

어떤 점에서 방향에 따라 기울기가 다른 함수(비등방성 함수)의 경우 탐색 경로가 비효율적임

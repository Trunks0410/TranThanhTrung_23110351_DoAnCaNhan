# ✨ĐỒ ÁN CÁ NHÂN MÔN TRÍ TUỆ NHÂN TẠO: Phân tích & ứng dụng các Thuật toán Tìm kiếm trên Bài toán 8 Ô Chữ✨

## 📚 Giới thiệu

Báo cáo này trình bày kết quả nghiên cứu và triển khai các thuật toán tìm kiếm cơ bản và nâng cao trong lĩnh vực Trí tuệ Nhân tạo. Dự án cá nhân này tập trung vào việc phân tích cơ chế hoạt động, đặc điểm lý thuyết và đánh giá hiệu quả hoạt động của các thuật toán này khi áp dụng trên bài toán kinh điển **8 ô chữ (8-Puzzle)**, một ví dụ điển hình cho bài toán tìm kiếm trạng thái. Nội dung báo cáo tóm tắt này được trình bày dựa trên cấu trúc yêu cầu của đề bài.

---

## 🎯 1. Mục tiêu

Các mục tiêu chính mà dự án hướng tới bao gồm:

* Nắm vững **cơ chế hoạt động** và **đặc điểm lý thuyết** của các nhóm thuật toán tìm kiếm đa dạng.
* Áp dụng thành công các thuật toán tìm kiếm vào việc giải **bài toán 8 ô chữ**.
* **Đánh giá định lượng và định tính** hiệu suất của các thuật toán (thời gian, bộ nhớ sử dụng, số lượng nút thăm) khi chạy trên 8 ô chữ thông qua các thực nghiệm.
* Tổng hợp và trình bày kết quả nghiên cứu một cách **trực quan và khoa học** trong báo cáo này.

---

## 📖 2. Nội dung Chi tiết

Phần này đi sâu vào từng nhóm thuật toán đã nghiên cứu, làm rõ cách chúng hoạt động và hiệu quả trên bài toán 8 ô chữ.

### 🧩 Bài toán 8 Ô Chữ: Định nghĩa Tìm kiếm

Trước khi đi vào từng thuật toán, hãy cùng định nghĩa các thành phần của bài toán 8 ô chữ dưới góc độ bài toán tìm kiếm:

* **Không gian trạng thái:** Tập hợp tất cả các cấu hình (cách sắp xếp) có thể có của 8 viên gạch số và 1 ô trống trên bảng 3x3.
* **Trạng thái ban đầu:** Cấu hình ban đầu của bảng (thường là ngẫu nhiên hoặc do người dùng cung cấp).
* **Hàm chuyển trạng thái:** Các hành động di chuyển ô trống (lên, xuống, trái, phải) nếu hợp lệ, dẫn đến cấu hình bảng mới (trạng thái mới).
* **Trạng thái đích:** Cấu hình bảng mà các ô số được sắp xếp theo thứ tự mong muốn.
* **Chi phí bước đi:** Thường là 1 cho mỗi nước đi.
* **Lời giải:** Một chuỗi các hành động hợp lệ từ trạng thái ban đầu dẫn đến trạng thái đích. Chi phí của lời giải là tổng chi phí các bước đi (bằng độ dài chuỗi khi chi phí = 1).

---

### 🔍 2.1. Các Thuật toán Tìm kiếm không có Thông tin (Uninformed Search)

Nhóm thuật toán này tìm kiếm mà không sử dụng thông tin về đích, chỉ dựa vào cấu trúc không gian trạng thái.

#### **Tìm kiếm theo chiều rộng (BFS)**

* **Cơ chế:** Khám phá không gian trạng thái theo từng cấp độ (level-by-level), sử dụng **Hàng đợi (Queue)**.
* **Đặc điểm:**
    * **Hoàn chỉnh (Complete):** Có.
    * **Tối ưu (Optimal):** Có, khi chi phí bước đi đồng nhất.
    * **Độ phức tạp thời gian:** $O(b^d)$.
    * **Độ phức tạp không gian:** $O(b^d)$ - nhược điểm lớn.
* **Áp dụng trên 8 ô chữ:** Tìm được lời giải ngắn nhất, nhưng yêu cầu bộ nhớ rất lớn với các bài toán sâu.

*Xem BFS hoạt động trên 8 ô chữ:*
![Minh họa GIF BFS 8-Puzzle](path/to/your/bfs_animation.gif)

#### **Tìm kiếm theo chiều sâu (DFS)**

* **Cơ chế:** Khám phá sâu nhất có thể theo một nhánh trước khi quay lui, sử dụng **Ngăn xếp (Stack)** hoặc đệ quy.
* **Đặc điểm:**
    * **Hoàn chỉnh:** Không (có thể đi vào vòng lặp hoặc nhánh vô hạn).
    * **Tối ưu:** Không.
    * **Độ phức tạp thời gian:** $O(b^m)$ ( $m$ là độ sâu lớn nhất).
    * **Độ phức tạp không gian:** $O(bm)$ - ưu điểm về bộ nhớ.
* **Áp dụng trên 8 ô chữ:** Tiết kiệm bộ nhớ hơn BFS, nhưng lời giải thường không tối ưu và có thể mất nhiều thời gian hơn.

*Xem DFS hoạt động trên 8 ô chữ:*
![Minh họa GIF DFS 8-Puzzle](path/to/your/dfs_animation.gif)

#### **Tìm kiếm chi phí đồng nhất (UCS)**

* **Cơ chế:** Mở rộng nút có chi phí đường đi từ gốc ($g(n)$) thấp nhất, sử dụng **Hàng đợi ưu tiên (Priority Queue)**.
* **Đặc điểm:**
    * **Hoàn chỉnh:** Có (với chi phí không âm).
    * **Tối ưu:** Có.
    * **Độ phức tạp:** Tương tự **BFS** khi chi phí bước đi bằng 1 ($O(b^d)$ thời gian/không gian).
* **Áp dụng trên 8 ô chữ:** Đảm bảo lời giải tối ưu chi phí (tương tự BFS), nhưng vẫn tốn bộ nhớ.

*Xem UCS hoạt động trên 8 ô chữ:*
![Minh họa GIF UCS 8-Puzzle](path/to/your/ucs_animation.gif)

#### **Tìm kiếm theo chiều sâu lặp sâu dần (IDS)**

* **Cơ chế:** Thực hiện chuỗi DFS với giới hạn độ sâu tăng dần (0, 1, 2, ...).
* **Đặc điểm:**
    * **Hoàn chỉnh:** Có.
    * **Tối ưu:** Có (khi chi phí đồng nhất).
    * **Độ phức tạp thời gian:** $O(b^d)$.
    * **Độ phức tạp không gian:** $O(bd)$ - ưu điểm bộ nhớ.
* **Áp dụng trên 8 ô chữ:** Cân bằng giữa tính tối ưu (như BFS) và hiệu quả bộ nhớ (như DFS), thường là lựa chọn tốt nhất trong nhóm không có thông tin cho 8 ô chữ.

*Xem IDS hoạt động trên 8 ô chữ:*
![Minh họa GIF IDS 8-Puzzle](path/to/your/ids_animation.gif)

---

*So sánh hiệu suất các thuật toán Tìm kiếm không có thông tin trên bài toán 8 ô chữ:*
![Biểu đồ so sánh Hiệu suất Uninformed Search 8-Puzzle](path/to/your/uninformed_performance_chart.png)

**Nhận xét về hiệu suất trên 8 ô chữ:** Dữ liệu thực nghiệm cho thấy **IDS** là thuật toán không có thông tin hiệu quả nhất trên 8 ô chữ. Mặc dù BFS/UCS tìm được lời giải tối ưu, yêu cầu bộ nhớ của chúng tăng lên rất nhanh với độ sâu lời giải. DFS tiết kiệm bộ nhớ nhưng lời giải thường dài hơn. IDS khắc phục được nhược điểm bộ nhớ của BFS mà vẫn giữ được tính tối ưu và hiệu quả thời gian tương đương.

---

### 🧠 2.2. Các Thuật toán Tìm kiếm có Thông tin (Informed Search / Heuristic Search)

Nhóm này sử dụng **Hàm Heuristic ($h(n)$)** - ước lượng chi phí từ trạng thái $n$ đến đích - để hướng dẫn quá trình tìm kiếm hiệu quả hơn. Các heuristic cho 8 ô chữ bao gồm:
* $h_1(n)$: Số ô sai vị trí.
* $h_2(n)$: Tổng khoảng cách Manhattan của các ô đến vị trí đích.

#### **Tìm kiếm tham lam nhất (Greedy Best-First Search)**

* **Cơ chế:** Luôn mở rộng nút mà heuristic ước lượng gần đích nhất ($h(n)$ nhỏ nhất).
* **Đặc điểm:**
    * **Hoàn chỉnh:** Không.
    * **Tối ưu:** Không.
* **Áp dụng trên 8 ô chữ:** Tìm kiếm nhanh chóng theo "trực giác" của heuristic, nhưng thường tìm thấy lời giải không tối ưu.

*Xem Greedy Search hoạt động trên 8 ô chữ:*
![Minh họa GIF Greedy Search 8-Puzzle](path/to/your/greedy_animation.gif)

#### **Thuật toán A* (A* Search)**

* **Cơ chế:** Mở rộng nút có hàm đánh giá $f(n) = g(n) + h(n)$ thấp nhất, cân bằng giữa chi phí đã đi ($g$) và ước lượng chi phí còn lại ($h$).
* **Đặc điểm:**
    * **Hoàn chỉnh:** Có.
    * **Tối ưu:** Có, nếu heuristic được chấp nhận ($h(n) \le h^*(n)$). $h_1$ và $h_2$ đều được chấp nhận.
    * **Độ phức tạp thời gian/không gian:** Phụ thuộc vào chất lượng heuristic, trường hợp xấu nhất là $O(b^d)$, nhưng thường hiệu quả hơn nhiều trong thực tế.
* **Áp dụng trên 8 ô chữ:** Là thuật toán hiệu quả nhất trong nhóm tìm kiếm trạng thái. Với heuristic **Manhattan Distance ($h_2$)**, A* tìm được lời giải tối ưu rất nhanh chóng.

*Xem A* hoạt động trên 8 ô chữ:*
![Minh họa GIF A* Search 8-Puzzle](path/to/your/a_star_animation.gif)

#### **Tìm kiếm theo chiều sâu lặp sâu dần với A* (IDA*)**

* **Cơ chế:** Phiên bản lặp sâu dần của A*, giới hạn tìm kiếm theo ngưỡng $f(n)$ thay vì độ sâu.
* **Đặc điểm:**
    * **Hoàn chỉnh:** Có.
    * **Tối ưu:** Có (nếu heuristic được chấp nhận).
    * **Độ phức tạp thời gian:** $O(b^d)$.
    * **Độ phức tạp không gian:** $O(bd)$ - vượt trội A* truyền thống về bộ nhớ.
* **Áp dụng trên 8 ô chữ:** Lựa chọn tối ưu khi bài toán lớn, đòi hỏi tính tối ưu của A* nhưng bộ nhớ là giới hạn.

*Xem IDA* hoạt động trên 8 ô chữ:*
![Minh họa GIF IDA* Search 8-Puzzle](path/to/your/ida_star_animation.gif)

---

*So sánh hiệu suất các thuật toán Tìm kiếm có thông tin trên bài toán 8 ô chữ (ví dụ với heuristic $h_2$):*
![Biểu đồ so sánh Hiệu suất Informed Search 8-Puzzle](path/to/your/informed_performance_chart.png)

**Nhận xét về hiệu suất trên 8 ô chữ:** Các thuật toán có thông tin, đặc biệt là **A*** và **IDA*** với heuristic **Manhattan Distance**, cho thấy hiệu quả vượt trội so với nhóm không có thông tin về tốc độ và số nút thăm trong khi vẫn đảm bảo tính tối ưu của lời giải. IDA* là giải pháp lý tưởng khi cần tối ưu cả về thời gian và bộ nhớ. Greedy Search nhanh nhưng không đáng tin cậy về tính tối ưu.

---

### ⛰️ 2.3. Các Thuật toán Tìm kiếm Cục bộ (Local Search)

Tìm kiếm cục bộ hoạt động bằng cách di chuyển giữa các trạng thái lân cận trong không gian cấu hình để tìm trạng thái tốt nhất theo một hàm mục tiêu. Chúng không lưu lại đường đi và thường được dùng cho các bài toán tối ưu hóa, không phải tìm đường đi như 8 ô chữ.

* **Một số thuật toán:** **Hill Climbing** (Simple, Steepest, Stochastic), **Simulated Annealing**, **Local Beam Search**, **Genetic Algorithm**.
* **Đặc điểm:** Rất hiệu quả về bộ nhớ, nhưng có nguy cơ mắc kẹt ở tối ưu cục bộ (trừ các biến thể như Simulated Annealing).

*Minh họa chung về ý tưởng Tìm kiếm Cục bộ:*
![Minh họa Tìm kiếm Cục bộ](path/to/your/local_search_illustration.png)

---

### 🔐 2.4. Các Thuật toán Tìm kiếm Ràng buộc (Constraint Satisfaction Problems - CSP)

Giải quyết bài toán bằng cách tìm gán giá trị cho các biến sao cho thỏa mãn tập hợp các ràng buộc. 8 ô chữ thường không được mô hình hóa như CSP để tìm đường đi.

* **Một số thuật toán:** **Generate and Test**, **Backtracking Search**, **AC3**.
* **Đặc điểm:** Phù hợp cho các bài toán gán giá trị, lập lịch, quy hoạch.

*Minh họa chung về ý tưởng giải CSP:*
![Minh họa CSP](path/to/your/csp_illustration.png)

---

### 🗺️ 2.5. Tìm kiếm trong Môi trường Phức tạp

Áp dụng cho các môi trường không hoàn toàn quan sát được hoặc có tính ngẫu nhiên. 8 ô chữ tiêu chuẩn là môi trường đơn giản, xác định.

* **Các kỹ thuật:** Sử dụng **Đồ thị AND-OR**, **Tìm kiếm không có quan sát**, **Tìm kiếm có quan sát một phần**.
* **Đặc điểm:** Cần thiết cho các bài toán phức tạp hơn trong thế giới thực.

*Minh họa chung về Tìm kiếm trong Môi trường Phức tạp:*
![Minh họa Tìm kiếm trong Môi trường Phức tạp](path/to/your/complex_search_illustration.png)

---

### 🤖 2.6. Tìm kiếm Học tăng cường (Reinforcement Learning - RL)

Tác nhân học cách đưa ra quyết định thông qua tương tác và nhận phản hồi (phần thưởng) từ môi trường. Có thể áp dụng cho 8 ô chữ như một bài toán RL, nhưng không phải cách tiếp cận thông thường.

* **Thuật toán chính được đề cập:** **Q-Learning** (học bảng giá trị Q cho cặp trạng thái-hành động).

*Minh họa chung về chu trình RL và Q-Learning:*
![Minh họa Q-Learning](path/to/your/q_learning_illustration.png)

---

## ✅ 3. Kết luận

Dự án này đã mang lại cho tôi cái nhìn tổng quan và sâu sắc về các thuật toán tìm kiếm trong AI, đặc biệt thông qua việc áp dụng chúng vào bài toán 8 ô chữ.

Dựa trên kết quả thực nghiệm, tôi xác nhận rằng các thuật toán tìm kiếm trạng thái như **A*** và **IDA*** (với heuristic Manhattan Distance) là những phương pháp **hiệu quả và tối ưu nhất** để giải bài toán 8 ô chữ. IDS là lựa chọn thay thế tốt trong nhóm không có thông tin. Các nhóm thuật toán khác có vai trò quan trọng nhưng phù hợp hơn với các loại bài toán khác trong lĩnh vực AI.

Việc thực hiện dự án này không chỉ củng cố kiến thức lý thuyết của tôi mà còn trang bị kỹ năng đánh giá hiệu suất thực tế của các thuật toán, làm nền tảng cho việc giải quyết các bài toán phức tạp hơn trong tương lai.

---

## 🧑‍💻 Tác giả

Trần Thành Trung - MSSV: 233110351
  
---

## 👨‍🏫 Giảng viên Hướng dẫn

GV: Phan Thị Huyền Trang

---

## 📚 Tài liệu Tham khảo

* Liệt kê các tài liệu (sách, bài báo, website,...) mà bạn đã tham khảo để thực hiện dự án này.
* Nên trình bày theo một định dạng nhất quán (ví dụ: Tên tác giả, Năm xuất bản, Tên tài liệu, Nhà xuất bản/Nguồn).



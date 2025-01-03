import { ComponentFixture, TestBed } from '@angular/core/testing';

import { PostFrequencyComponent } from './post-frequency.component';

describe('PostFrequencyComponent', () => {
  let component: PostFrequencyComponent;
  let fixture: ComponentFixture<PostFrequencyComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [PostFrequencyComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(PostFrequencyComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
